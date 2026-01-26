import math
import logging
from typing import List, Tuple
import torch
from colorama import Fore, Style, init

from src.models.logic_rag import LogicRAG

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicRAGPruned(LogicRAG):
    def __init__(
        self,
        corpus_path: str = None,
        cache_dir: str = "./cache",
        filter_repeats: bool = False,
        prune_ratio: float = 0.2,
        min_dependencies: int = 1,
    ):
        super().__init__(corpus_path, cache_dir, filter_repeats)
        self.MODEL_NAME = "LogicRAGPruned"
        self.prune_ratio = prune_ratio
        self.min_dependencies = min_dependencies

    def _prune_dependencies(self, question: str, dependencies: List[str]) -> List[str]:
        if not dependencies:
            return dependencies

        keep_count = max(self.min_dependencies, math.ceil(len(dependencies) * self.prune_ratio))
        if keep_count >= len(dependencies):
            return dependencies

        try:
            with torch.no_grad():
                q_emb = self.model.encode([question], convert_to_tensor=True)[0].cpu()
                dep_embs = self.model.encode(dependencies, convert_to_tensor=True).cpu()

            sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), dep_embs)
            top_scores, top_indices = torch.topk(sims, k=keep_count)
            kept = set(top_indices.tolist())

            pruned = [dep for i, dep in enumerate(dependencies) if i in kept]
            logger.info(
                f"Selective pruning kept {len(pruned)}/{len(dependencies)} dependencies"
            )
            return pruned
        except Exception as e:
            logger.error(f"{Fore.RED}Error pruning dependencies: {e}{Style.RESET_ALL}")
            return dependencies

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        info_summary = ""
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []
        dependency_analysis_history = []
        retrieved_chunks_set = set() if self.filter_repeats else None

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        #===============================================
        #== Stage 1: warm up retrieval ==
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)
        last_contexts = new_contexts
        info_summary = self.refine_summary_with_context(
            question,
            new_contexts,
            info_summary
        )

        analysis = self.warm_up_analysis(question, info_summary)

        if analysis["can_answer"]:
            print("Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, info_summary)
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count
        else:
            logger.info("Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
            pruned_dependencies = self._prune_dependencies(question, sorted_dependencies)
            dependency_analysis_history.append(
                {"sorted_dependencies": sorted_dependencies, "pruned_dependencies": pruned_dependencies}
            )
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
            logger.info(f"Pruned dependencies: {pruned_dependencies}\n\n")

        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0

        while round_count < self.max_rounds and idx < len(pruned_dependencies):
            round_count += 1

            current_query = pruned_dependencies[idx]
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts

            info_summary = self.refine_summary_with_context(
                question,
                new_contexts,
                info_summary
            )

            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")

            analysis = self.dependency_aware_rag(question, info_summary, pruned_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            })

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                answer = self.generate_answer(question, info_summary)
                self.last_dependency_analysis = dependency_analysis_history
                return answer, last_contexts, round_count
            else:
                idx += 1

        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count
