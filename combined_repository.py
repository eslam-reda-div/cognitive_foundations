"""
Combined Repository File: cognitive_foundations
========================================================
This file contains all the code and text content from the cognitive_foundations repository
combined into a single file without any changes to the original logic.

Repository Structure:
- reasoning_structure/ - Core Python modules for reasoning analysis
- element_annotation/ - Annotation guidelines and prompts
- structure_guidance/ - Guidance templates and prompt templates
- construct_graphs.py - Main entry point for graph construction

"""

# =============================================================================
# FILE: construct_graphs.py
# =============================================================================

import os
import sys
from collections import defaultdict
import argparse
from reasoning_structure.structure import SpanTree
from reasoning_structure.subgraph import ConsensusTreeFinder
from structure_guidance.generate_steered_traces import ElementGuidedReasoning

id2type = {
    -1: "No Type Matched",
    10: "Logical",
    1: "Algorithmic",
    2: "Story Problem",
    3: "Rule-Using", 
    4: "Decision-Making",
    5: "Troubleshooting",
    6: "Diagnosis-Solution",
    7: "Strategic Performance",
    8: "Case Analysis",
    9: "Design",
    11: "Dilemma",
    12: "Factual Recall/Comprehension",
    13: "Creative/Expressive"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_dir", type=str, default='/shared/data3/pk36/structured_survey/span_example_annotation', help="Directory containing element files")
    parser.add_argument("--prompt_template_dir", type=str, default='structure_guidance/prompt_templates', help="Directory containing outputted templates")
    parser.add_argument("--overlap_threshold", type=float, default=0.8, help="Overlap threshold for span tree")
    parser.add_argument("--parallel_threshold", type=float, default=20, help="Parallel threshold for span tree")
    parser.add_argument("--target_type", type=str, default=None, help="Target type to filter elements")
    parser.add_argument("--output_dir", type=str, default="reasoning_structure/output_consensus_graphs", help="Output directory for consensus graphs")
    parser.add_argument("--path_to_question_info", type=str, default="/shared/data3/pk36/structured_survey/accuracy_evaluation/all_evaluations_gpt4o.json", help="Directory for question info")
    parser.add_argument("--max_nodes", type=int, default=7, help="Maximum number of nodes in consensus graph")
    parser.add_argument("--generate_steered_traces", action="store_true")
    args = parser.parse_args()

    tree = SpanTree(
        overlap_threshold=args.overlap_threshold,
        parallel_threshold=args.parallel_threshold
    )

    # Load span-level annotation files

    if not os.path.exists(args.element_dir):
        print(f"Element directory with span-level annotations ({args.element_dir}) does not exist.")
        sys.exit(1)
    
    # Create template output directory if it does not exist
    if not os.path.exists(args.prompt_template_dir):
        os.makedirs(args.prompt_template_dir)

    tree.load_element_files(args.element_dir, id2type=id2type, target_type=args.target_type)

    print("Data loaded successfully!")

    # Find consensus and success-prone subgraphs for each target problem type

    consensus_finder = ConsensusTreeFinder(span_tree=tree)
    type2graph = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Generate consensus and success-prone subgraphs for all target problem types
    problem_types = list(id2type.values())
    if "No Type Matched" in problem_types:
        problem_types.remove("No Type Matched")

    target_problem_types = problem_types[1:-2]
    target_problem_types.remove('Strategic Performance') # not applicable for non-real-world settings

    success_only = True
    success_suffix = 'success' if success_only else 'all'

    for p_type in target_problem_types:
        print(f"Processing {p_type}...")
        core_graph = consensus_finder.construct_consensus_tree(p_type, max_nodes=args.max_nodes, success_only=success_only, dynamic=True)
        consensus_finder.visualize_semantic_consensus_graph(graph=core_graph, problem_type=p_type, max_nodes=args.max_nodes, 
                                                            output_file=f'{args.output_dir}/{args.max_nodes}/{success_suffix}_graph_{p_type}_{args.max_nodes}.png')
        type2graph[p_type][args.max_nodes][success_suffix] = core_graph

    success_only = False
    success_suffix = 'success' if success_only else 'all'

    for p_type in target_problem_types:
        print(f"Processing {p_type}...")
        core_graph = consensus_finder.construct_consensus_tree(p_type, max_nodes=args.max_nodes, success_only=success_only, dynamic=True)
        consensus_finder.visualize_semantic_consensus_graph(graph=core_graph, problem_type=p_type, max_nodes=args.max_nodes, 
                                                            output_file=f'{args.output_dir}/{args.max_nodes}/{success_suffix}_graph_{p_type}_{args.max_nodes}.png')
        type2graph[p_type][args.max_nodes][success_suffix] = core_graph

    print("Consensus + success-prone graphs all generated and saved successfully!")

    ## Linearize each success-prone graph
    guided_reasoning = ElementGuidedReasoning(args=args, span_tree=tree, consensus_finder=consensus_finder, problem_type_graphs=type2graph)
    
    type2linear = defaultdict(lambda: defaultdict()) # problem_type -> max_nodes : template
    for p_type, graphs in type2graph.items():
        for max_nodes, graph in graphs.items():
            type2linear[p_type][max_nodes] = guided_reasoning.graph_to_prompt(graph['success'])
    
    with open('structure_guidance/template_prompt.txt', 'r') as f:
        prompt_template = f.read()
    
    ## Convert linearized graph into prompt (using structure_guidance/template_prompt.txt)
    for p_type in type2linear:
        for node_num in type2linear[p_type]:
            with open(f'{args.prompt_template_dir}/{p_type}_{node_num}.txt', 'w') as f:
                prompt_text = prompt_template.format(p_type=p_type, node_num=node_num, structure_info=type2linear[p_type][node_num])
                f.write(prompt_text)

# =============================================================================
# FILE: reasoning_structure/structure.py
# =============================================================================

import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json_repair
from tqdm import tqdm
from scipy import stats
import numpy as np
import math


@dataclass
class Model:
    """Represents a model and tracks problems it has answered."""
    name: str
    problems: Set[str] = field(default_factory=set)
    correct_problems: Set[str] = field(default_factory=set)
    incorrect_problems: Set[str] = field(default_factory=set)
    
    def add_problem(self, problem_id: str, is_correct: bool):
        """Add a problem that this model has answered."""
        self.problems.add(problem_id)
        if is_correct:
            self.correct_problems.add(problem_id)
        else:
            self.incorrect_problems.add(problem_id)
    
    def get_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.problems:
            return 0.0
        return len(self.correct_problems) / len(self.problems)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name
        return False


@dataclass
class Problem:
    """Represents a problem/question with its metadata."""
    problem_id: str
    task_category: str = ""
    problem_type: str = ""
    modality: str = "text"
    correctness: Dict[str, bool] = field(default_factory=dict)  # model_name -> is_correct
    
    def add_model_result(self, model_name: str, is_correct: bool):
        """Record a model's correctness on this problem."""
        self.correctness[model_name] = is_correct
    
    def get_correct_models(self) -> List[str]:
        """Get list of models that answered correctly."""
        return [m for m, correct in self.correctness.items() if correct]
    
    def get_incorrect_models(self) -> List[str]:
        """Get list of models that answered incorrectly."""
        return [m for m, correct in self.correctness.items() if not correct]
    
    def __hash__(self):
        return hash(self.problem_id)
    
    def __eq__(self, other):
        if isinstance(other, Problem):
            return self.problem_id == other.problem_id
        return False


@dataclass
class Edge:
    """Represents a connection between two element nodes."""
    node_a: str
    node_b: str
    edge_type: str  # "next", "contains", or "parallel"
    weight: int = 0
    occurrences: List[Tuple[str, str]] = field(default_factory=list)  # (model_name, problem_id)
    
    def __hash__(self):
        return hash((self.node_a, self.node_b, self.edge_type))
    
    def add_occurrence(self, model_name: str, problem_id: str):
        """Record an occurrence of this edge in a specific model's trace for a problem."""
        self.occurrences.append((model_name, problem_id))
        self.weight += 1
    
    def get_occurrences_by_model(self, model_name: str) -> int:
        """Count occurrences for a specific model."""
        return sum(1 for m, _ in self.occurrences if m == model_name)
    
    def get_occurrences_by_problem(self, problem_id: str) -> int:
        """Count occurrences for a specific problem."""
        return sum(1 for _, p in self.occurrences if p == problem_id)


@dataclass
class ElementNode:
    """Represents a element node with its spans and connections."""
    element: str
    spans: List[Tuple[int, int, str, str]] = field(default_factory=list)  # (start, end, model_name, problem_id)
    total_span_length: int = 0
    frequency: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))  # (model, problem) -> count
    connections: Dict[str, Dict[str, List[Edge]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    def add_span(self, start: int, end: int, model_name: str, problem_id: str):
        """Add a span to this element node."""
        self.spans.append((start, end, model_name, problem_id))
        self.total_span_length += (end - start)
        self.frequency[(model_name, problem_id)] += 1
    
    def get_total_occurrences(self) -> int:
        """Get total number of occurrences across all model-problem pairs."""
        return sum(self.frequency.values())
    
    def get_avg_span_length(self) -> float:
        """Calculate average span length."""
        if not self.spans:
            return 0.0
        return self.total_span_length / len(self.spans)
    
    def add_connection(self, other_element: str, edge: Edge):
        """Add a connection to another element."""
        self.connections[other_element][edge.edge_type].append(edge)


class SpanTree:
    """Builds and manages a DAG of element relationships."""
    
    def __init__(self, overlap_threshold: float = 0.8, parallel_threshold: int = 20):
        """
        Initialize SpanTree.
        
        Args:
            overlap_threshold: Minimum overlap ratio for "parallel" relationship (0.0-1.0)
        """
        self.nodes: Dict[str, ElementNode] = {}
        # edges is a nested dict: (node_a, node_b) -> {edge_type: Edge}
        self.edges: Dict[Tuple[str, str], Dict[str, Edge]] = defaultdict(dict)
        self.overlap_threshold = overlap_threshold
        self.parallel_threshold = parallel_threshold
        self.trace_elements: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = defaultdict(list)  # (model, problem) -> elements
        
        self.models: Dict[str, Model] = {}
        self.problems: Dict[str, Problem] = {}
    
    def get_all_edges(self) -> List[Edge]:
        """Get a flat list of all edges (for backward compatibility)."""
        edges = []
        for edge_types in self.edges.values():
            edges.extend(edge_types.values())
        return edges
    
    def get_edge(self, node_a: str, node_b: str, edge_type: str) -> Optional[Edge]:
        """Get a specific edge by its nodes and type."""
        edge_key = (node_a, node_b)
        if edge_key in self.edges:
            return self.edges[edge_key].get(edge_type)
        return None
    
    def load_element_files(self, directory: str, id2type: dict, target_type: str = None):
        """Load all element JSON files from a directory in parallel."""
        json_files = glob.glob(os.path.join(directory, '*.json'))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {directory}")
        
        print(f"Loading {len(json_files)} element files in parallel...")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()*1.5) as executor:
            future_to_file = {
                executor.submit(
                    self._process_single_file, 
                    json_file,
                    id2type, 
                    target_type
                ): json_file 
                for json_file in json_files
            }
            
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        model_name, trace_elements, node_data, problems_data, models_data = result
                        
                        # Thread-safe: Consolidate in main thread
                        
                        # 1. Add/update problems
                        for problem_id, problem_info in problems_data.items():
                            if problem_id not in self.problems:
                                self.problems[problem_id] = Problem(
                                    problem_id=problem_id,
                                    task_category=problem_info['task_category'],
                                    problem_type=problem_info['problem_type'],
                                    modality=problem_info['modality']
                                )
                            # Add model result to problem
                            for model, is_correct in problem_info['model_results'].items():
                                self.problems[problem_id].add_model_result(model, is_correct)
                        
                        # 2. Add/update models
                        for model, model_info in models_data.items():
                            if model not in self.models:
                                self.models[model] = Model(name=model)
                            # Add problem results to model
                            for prob_id, is_correct in model_info['problem_results'].items():
                                self.models[model].add_problem(prob_id, is_correct)
                        
                        # 3. Add trace elements
                        for trace_key, elements in trace_elements.items():
                            self.trace_elements[trace_key].extend(elements)
                        
                        # 4. Add node data
                        for element_name, spans in node_data.items():
                            if element_name not in self.nodes:
                                self.nodes[element_name] = ElementNode(element=element_name)
                            node = self.nodes[element_name]
                            for start, end, model, problem in spans:
                                node.add_span(start, end, model, problem)
                        
                        print(f"  ✓ Loaded {model_name}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {json_file}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Loaded {len(self.nodes)} elements")
        print(f"Loaded {len(self.models)} models")
        print(f"Loaded {len(self.problems)} problems")
        print(f"Loaded {len(self.trace_elements)} traces")
        self._build_relationships()


    def _process_single_file(self, json_file: str, id2type: dict, target_type: str = None) -> Optional[Tuple]:
        """
        Process a single model file.
        
        Returns all data WITHOUT modifying shared state (thread-safe).
        """
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json_repair.load(f)
        
        trace_elements = defaultdict(list)
        node_data = defaultdict(list)  # element_name -> [(start, end, model, problem), ...]
        problems_data = {}  # problem_id -> {task_category, problem_type, modality, model_results}
        models_data = defaultdict(lambda: {'problem_results': {}})  # model -> {problem_results: {prob_id: is_correct}}
        
        model_name = None  # Will be set from first question
        
        for question_key, question_data in data.items():
            #try:
            problem_id = str(question_data.get('question_id', question_key))
            if ('hier_' in problem_id) or ('graph_' in problem_id):
                continue
            current_model_name = question_data['model_name']
            
            # Set model_name on first iteration
            if model_name is None:
                model_name = current_model_name
            # Verify all questions in file are from same model
            elif model_name != current_model_name:
                print(f"  ⚠ Warning: Multiple models in {json_file}: {model_name} vs {current_model_name}")
                # Use the first model name seen
            
            task_category = question_data['task']

            problem_types = question_data['problem_type']
            if problem_types is None:
                problem_type = None
            elif type(problem_types) == str and not problem_types.isnumeric():
                problem_type = problem_types
            else:
                info = [int(i) if i.isnumeric() else -1 for i in problem_types]
                problem_type = id2type[stats.mode(np.array(info)).mode]

            # Filter by target type if specified
            if target_type and (problem_type != target_type):
                continue

            modality = question_data.get('modality', 'text')
            if 'image' in json_file.lower():
                correctness = True
                modality = question_data['correctness']
            elif 'audio' in json_file.lower():
                correctness = True
                modality = "audio"
            else:
                correctness = question_data['correctness']

            # Store problem data (will be created/updated in main thread)
            if problem_id not in problems_data:
                problems_data[problem_id] = {
                    'task_category': task_category,
                    'problem_type': problem_type,
                    'modality': modality,
                    'model_results': {}
                }
            problems_data[problem_id]['model_results'][model_name] = correctness
            
            # Store model data (will be created/updated in main thread)
            models_data[model_name]['problem_results'][problem_id] = correctness

            # Process element annotations
            for element_name, element_info in question_data['element_annotation'].items():
                # Only consider if score >= 2
                if ('score' not in element_info) or (element_info['score'] < 2):
                    continue

                for span_group in element_info['spans']:
                    if not isinstance(span_group, list) or len(span_group) != 2:
                        continue

                    start, end = span_group

                    # Validate span values
                    if isinstance(start, str):
                        if not start.isnumeric():
                            continue
                        start = int(start)
                    if isinstance(end, str):
                        if not end.isnumeric():
                            continue
                        end = int(end)
                    
                    # Store node data for main thread to process
                    node_data[element_name].append((start, end, model_name, problem_id))
                    
                    # Store trace element
                    trace_key = (model_name, problem_id)
                    trace_elements[trace_key].append((element_name, start, end))
            
            # except Exception as e:
            #     print(f"  ✗ Error processing question {question_key} in {json_file}: {e}")
            #     continue
        
        if model_name is None:
            return None
        
        return (model_name, trace_elements, node_data, problems_data, models_data)


    def _build_relationships(self):
        """Build relationships between elements based on span positions."""
        print("\nBuilding relationships in parallel...")

        trace_keys = list(self.trace_elements.keys())

        all_edges = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()*1.5) as executor:
            futures = {
                executor.submit(self._process_trace_relationships, trace_key): trace_key 
                for trace_key in trace_keys
            }
            
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing traces")
                
            for future in iterator:
                try:
                    edges = future.result()
                    all_edges.extend(edges)
                except Exception as e:
                    trace_key = futures[future]
                    print(f"  ✗ Error processing trace {trace_key}: {e}")
                    import traceback
                    traceback.print_exc()

        # Consolidate edges using the new dictionary structure
        for edge_data in all_edges:
            element_a, element_b, edge_type, model_name, problem_id = edge_data
            edge_key = (element_a, element_b)

            # Check if this edge type already exists
            if edge_type not in self.edges[edge_key]:
                # Create new edge
                edge = Edge(node_a=element_a, node_b=element_b, edge_type=edge_type)
                self.edges[edge_key][edge_type] = edge
            else:
                # Use existing edge
                edge = self.edges[edge_key][edge_type]

            edge.add_occurrence(model_name, problem_id)

            # Add connections to nodes
            self.nodes[element_a].add_connection(element_b, edge)
            if edge_type != "next":
                self.nodes[element_b].add_connection(element_a, edge)

        total_edges = sum(len(edge_types) for edge_types in self.edges.values())
        total_occurrences = sum(edge.weight for edge_types in self.edges.values() for edge in edge_types.values())
        
        print(f"Created {len(self.edges)} unique directed edges")
        print(f"Created {total_edges} unique edge types")
        print(f"Total edge occurrences: {total_occurrences}")


    def _process_trace_relationships(self, trace_key: Tuple[str, str]) -> List[Tuple]:
        """Process relationships for a single trace."""
        model_name, problem_id = trace_key
        elements_spans = self.trace_elements[trace_key]
        
        # Sort by start position, then by span length (descending)
        sorted_spans = sorted(elements_spans, key=lambda x: (x[1], -(x[2] - x[1])))
        
        edges = []
        for i, (element_a, start_a, end_a) in enumerate(sorted_spans):
            has_next = False
            for j, (element_b, start_b, end_b) in enumerate(sorted_spans):
                if i >= j:
                    continue
                
                edge_type = self._determine_relationship(start_a, end_a, start_b, end_b)
                
                # Refine "next" edges to only edges between a and b which have no other intermediate 'next' edge that is separate

                if edge_type == "next":
                    if has_next and (start_b >= end_a):
                        between = True
                        break
                    if has_next:
                        between = True
                        edge_type = None
                        break
                    else:
                        has_next = True
                        between = False
                
                if edge_type:
                    edges.append((element_a, element_b, edge_type, model_name, problem_id))
        
        return edges


    def _determine_relationship(self, start_a: int, end_a: int, start_b: int, end_b: int) -> Optional[str]:
        """Determine the relationship type between two spans."""
        if ((end_b - start_b) == 0) or ((end_a - start_a) == 0):
            return None

        if (abs(start_b - start_a) + abs(end_b - end_a)) <= self.parallel_threshold:
            return "parallel"

        if start_b <= end_a:
            if end_a <= end_b:
                overlap = (end_a - start_b)/(end_b - start_b)
                if overlap >= self.overlap_threshold:
                    return "contains"
                else:
                    return "next"
            else: # end_a > end_b
                overlap = (end_b - start_b) / (end_a - start_a)
                if overlap >= self.overlap_threshold:
                    return "parallel"
                else:
                    return "next"
        else: # end_a < start_b
            return "next"
    

    # ===== ANALYTICAL FUNCTIONS =====

    def compute_npmi_node(self, problem_type: str = None, all_elements: list = None) -> Dict:
        """
        Compute normalized pointwise mutual information (NPMI) between each element 
        and successful reasoning traces.
        
        NPMI(element, success) = PMI(element, success) / -log(P(element, success))
        where PMI(element, success) = log(P(element, success) / (P(element) * P(success)))
        
        NPMI ranges from -1 to 1:
        - 1: perfect positive association (element always appears with success)
        - 0: independence (element and success are unrelated)
        - -1: perfect negative association (element never appears with success)
        
        Args:
            problem_type: Filter by specific problem type (None for all)
            
        Returns:
            Dictionary with NPMI scores and component probabilities for each element
        """
        element_and_success = Counter({b: 0 for b in all_elements})  # Count of traces with both element and success
        element_only = Counter({b: 0 for b in all_elements})  # Count of traces with element (success or not)
        success_count = 0
        total_traces = 0
        
        for (model_name, problem_id), elements in self.trace_elements.items():
            if problem_id not in self.problems:
                continue
            
            problem = self.problems[problem_id]
            
            # Filter by problem type if specified
            if problem_type and problem.problem_type != problem_type:
                continue
            
            total_traces += 1
            is_correct = problem.correctness.get(model_name, False)
            
            if is_correct:
                success_count += 1
            
            element_names = set([b[0] for b in elements])
            
            # Count element occurrences
            element_only.update(element_names)
            
            # Count joint occurrences (element AND success)
            if is_correct:
                element_and_success.update(element_names)
        
        if total_traces == 0:
            return {}
        
        # Compute probabilities and NPMI
        p_success = success_count / total_traces
        
        npmi_scores = {}
        
        for element in all_elements:
            p_element = element_only[element] / total_traces
            p_element_and_success = element_and_success.get(element, 0) / total_traces
            
            # Avoid log(0) errors
            if p_element_and_success == 0 or p_element == 0 or p_success == 0:
                npmi_scores[element] = {
                    'npmi': -1.0,  # Minimum value indicates no co-occurrence
                    'pmi': float('-inf'),
                    'p_element': p_element,
                    'p_success': p_success,
                    'p_element_and_success': p_element_and_success
                }
                continue
            
            # PMI = log2(P(element, success) / (P(element) * P(success)))
            pmi = math.log2(p_element_and_success / (p_element * p_success))
            
            # NPMI = PMI / -log2(P(element, success))
            # Normalization ensures NPMI is in [-1, 1]
            if (-math.log2(p_element_and_success)) == 0:
                npmi = 0
            else:
                npmi = pmi / (-math.log2(p_element_and_success))
            
            npmi_scores[element] = {
                'npmi': npmi,
                'pmi': pmi,
                'p_element': p_element,
                'p_success': p_success,
                'p_element_and_success': p_element_and_success,
                'count_element': element_only[element],
                'count_success_with_element': element_and_success.get(element, 0)
            }
        
        return npmi_scores


    def compute_npmi_edge(self, problem_type: str = None) -> Dict:
        """
        Compute normalized pointwise mutual information (NPMI) between each edge 
        and successful reasoning traces.
        
        Uses pre-built edges from SpanTree.edges instead of rebuilding them.
        
        NPMI(edge, success) = PMI(edge, success) / -log(P(edge, success))
        where PMI(edge, success) = log(P(edge, success) / (P(edge) * P(success)))
        
        NPMI ranges from -1 to 1:
        - 1: perfect positive association (edge always appears with success)
        - 0: independence (edge and success are unrelated)
        - -1: perfect negative association (edge never appears with success)
        
        Args:
            problem_type: Filter by specific problem type (None for all)
            
        Returns:
            Dictionary mapping (node_a, node_b, edge_type) -> NPMI statistics
        """
        edge_and_success = Counter()  # Count of traces with both edge and success
        edge_only = Counter()  # Count of traces with edge (success or not)
        success_count = 0
        total_traces = 0
        
        # Build a set of valid traces for this problem type
        valid_traces = set()
        
        for (model_name, problem_id) in self.trace_elements.keys():
            if problem_id not in self.problems:
                continue
            
            problem = self.problems[problem_id]
            
            # Filter by problem type if specified
            if problem_type and problem.problem_type != problem_type:
                continue
            
            valid_traces.add((model_name, problem_id))
            total_traces += 1
            
            is_correct = problem.correctness.get(model_name, False)
            if is_correct:
                success_count += 1
        
        if total_traces == 0:
            return {}
        
        # Use pre-built edges from SpanTree
        for edge_key, edge_types in self.edges.items():
            node_a, node_b = edge_key
            
            for edge_type, edge in edge_types.items():
                # Count occurrences in valid traces
                trace_key = (node_a, node_b, edge_type)
                
                for model_name, problem_id in edge.occurrences:
                    # Skip if not in our filtered set
                    if (model_name, problem_id) not in valid_traces:
                        continue
                    
                    edge_only[trace_key] += 1
                    
                    # Check if this trace was successful
                    if self.problems[problem_id].correctness.get(model_name, False):
                        edge_and_success[trace_key] += 1
        
        # Compute probabilities and NPMI
        p_success = success_count / total_traces
        
        npmi_scores = {}
        
        for edge_key in edge_only.keys():
            p_edge = edge_only[edge_key] / total_traces
            p_edge_and_success = edge_and_success.get(edge_key, 0) / total_traces
            
            # Avoid log(0) errors
            if p_edge_and_success == 0 or p_edge == 0 or p_success == 0:
                npmi_scores[edge_key] = {
                    'npmi': -1.0,  # Minimum value indicates no co-occurrence
                    'pmi': float('-inf'),
                    'p_edge': p_edge,
                    'p_success': p_success,
                    'p_edge_and_success': p_edge_and_success
                }
                continue
            
            # PMI = log2(P(edge, success) / (P(edge) * P(success)))
            pmi = math.log2(p_edge_and_success / (p_edge * p_success))

            if (-math.log2(p_edge_and_success)) == 0:
                npmi = 0
            else:
                # NPMI = PMI / -log2(P(edge, success))
                npmi = pmi / (-math.log2(p_edge_and_success))
            
            npmi_scores[edge_key] = {
                'npmi': npmi,
                'pmi': pmi,
                'p_edge': p_edge,
                'p_success': p_success,
                'p_edge_and_success': p_edge_and_success,
                'count_edge': edge_only[edge_key],
                'count_success_with_edge': edge_and_success.get(edge_key, 0)
            }
        
        return npmi_scores

# =============================================================================
# FILE: reasoning_structure/subgraph.py
# =============================================================================

import os
import matplotlib.pyplot as plt
import networkx as nx
import colorsys
import pygraphviz as pgv
from collections import Counter
from typing import List, Tuple


# ----------------------------------------------------------------------------------------------------------------------
# Methods for identifying the most common/consensus or success-prone tree structure
# from multiple successful reasoning traces.

class ConsensusTreeFinder:
    """Find consensus element tree structures from multiple successful traces."""
    
    def __init__(self, span_tree):
        """
        Initialize with a SpanTree instance.
        
        Args:
            span_tree: SpanTree instance with loaded data
        """
        self.span_tree = span_tree

    def construct_consensus_tree(self, problem_type: str, max_nodes: int = 10, success_only: bool = True, dynamic: bool = False) -> nx.DiGraph:
        """
        Construct a consensus tree of elements for a given problem type based on NPMI.
        
        The algorithm:
        1. Find the element with highest NPMI that appears first (earliest start) in successful traces
        2. From current element A, find the edge (A->B) with highest NPMI (any type: contains/next/parallel)
        3. Add edge to graph and move to element B
        4. Repeat until max_nodes reached or no more valid edges
        
        Args:
            problem_type: The problem type to analyze
            max_nodes: Maximum number of nodes to include in the consensus tree
            
        Returns:
            NetworkX DiGraph representing the consensus reasoning path
        """
        # Compute NPMI for elements and edges
        element_npmi = self.span_tree.compute_npmi_node(problem_type=problem_type, 
                                        all_elements=list(self.span_tree.nodes.keys()))
        edge_npmi = self.span_tree.compute_npmi_edge(problem_type=problem_type)
        
        if not element_npmi or not edge_npmi:
            print("No NPMI data available")
            return nx.DiGraph()
        
        # Get successful traces for this problem type
        traces = self._get_all_traces_type(problem_type=problem_type, success_only=success_only)
        
        if not traces:
            print("No traces found")
            return nx.DiGraph()
        
        # Find elements that appear first (earliest start) in successful traces
        first_elements = Counter()
        
        for trace_key in traces:
            if trace_key not in self.span_tree.trace_elements:
                continue
            
            elements = self.span_tree.trace_elements[trace_key]
            if not elements:
                continue
            
            # Sort by start position and get the first element
            sorted_elements = sorted(elements, key=lambda x: (x[1], -(x[2]-x[1])))
            first_element = sorted_elements[0][0]
            first_elements[first_element] += 1
        
        if not first_elements:
            print("No first elements found")
            return nx.DiGraph()
        
        if success_only:
            # Pick the first element with highest NPMI
            candidate_first = [(b, element_npmi[b]['npmi']) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        else:
            # Pick the first element with highest count
            candidate_first = [(b, first_elements[b]) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        
        if not candidate_first:
            print("No valid first elements with NPMI scores")
            return nx.DiGraph()
        
        # Start with the element that has highest NPMI (or overall frequency if all traces) among first elements
        current_element = max(candidate_first, key=lambda x: x[1])[0]
        
        # Build consensus graph
        consensus_graph = nx.DiGraph()
        consensus_graph.add_node(current_element, 
                                npmi=element_npmi[current_element]['npmi'],
                                is_start=True)
        
        visited = {current_element}
        
        print(f"Starting consensus tree with: {current_element} (NPMI: {element_npmi[current_element]['npmi']:.3f}, Node Prob: {element_npmi[current_element]['p_element']:.3f})")
        
        # Iteratively add nodes based on highest NPMI edges
        while (len(consensus_graph.nodes()) < max_nodes):
            # Find all outgoing edges from current element
            candidate_edges = []
            
            for (node_a, node_b, edge_type), npmi_data in edge_npmi.items():
                # Must originate from current element
                if node_a != current_element:
                    continue
                
                # Skip self-loops
                if node_a == node_b:
                    continue
                
                # Skip if target already visited (avoid cycles)
                if node_b in visited:
                    continue
                
                candidate_edges.append((node_b, edge_type, npmi_data['npmi'], npmi_data['p_edge']))
            
            if not candidate_edges:
                print(f"No more valid edges from {current_element}")
                break
            
            # Pick edge with highest NPMI
            if success_only:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[2])
            else:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[3])

            if success_only and dynamic and (edge_npmi_score <= 0):
                break
            
            # Add to graph
            consensus_graph.add_node(next_element, 
                                    npmi=element_npmi[next_element]['npmi'],
                                    p_element=element_npmi[next_element]['p_element'])
            consensus_graph.add_edge(current_element, next_element, 
                                    edge_type=edge_type,
                                    npmi=edge_npmi_score,
                                    p_edge=edge_prob_score)
            
            print(f"  Added edge: {current_element} --[{edge_type}]--> {next_element} "
                  f"(Edge NPMI: {edge_npmi_score:.3f}, Edge Prob: {edge_prob_score:.3f}, Node NPMI: {element_npmi[next_element]['npmi']:.3f}, Node Prob: {element_npmi[next_element]['p_element']:.3f}")
            
            visited.add(next_element)
            current_element = next_element
        
        print(f"\nConsensus tree complete: {len(consensus_graph.nodes())} nodes, {len(consensus_graph.edges())} edges")
        
        return consensus_graph

    def _get_all_traces_type(self, problem_type: str, success_only: bool = True) -> List[Tuple[str, str]]:
        """Get all successful (model, problem) traces for a problem type."""

        all_traces = []

        for problem_id, problem_info in self.span_tree.problems.items():
            if problem_info.problem_type != problem_type:
                continue
            
            if success_only:
                models = problem_info.get_correct_models()
            else:
                models = list(problem_info.correctness.keys())
            
            for model_name in models:
                if (model_name, problem_id) in self.span_tree.trace_elements:
                    all_traces.append((model_name, problem_id))
        
        return all_traces
    
    def visualize_semantic_consensus_graph(self,
                                           graph: nx.DiGraph,
                                           problem_type: str,
                                           max_nodes: int,
                                           output_file: str = "consensus_tree.png",
                                           font_name: str = "Nimbus Sans") -> None:
        """
        Improved visualization for reasoning-element graphs.
        - Adds a title and legend.
        - Uses consistent, categorical colors for element nodes.
        - Sequential edges define vertical flow (top → bottom).
        - Modern aesthetic with light fill and dark borders/fonts.
        """

        # --- 1. Setup & Directory ---
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if graph.number_of_nodes() == 0:
            print("Empty graph.")
            return

        # --- 2. Define Element Color Map (Consistent Coloring) ---
        all_elements = [
            'self-awareness', 'self-evaluation', 'logical-coherence', 
            'compositionality', 'sequential-organization', 'selective-attention', 
            'forward-chaining', 'causal-organization', 'knowledge-structure-alignment', 
            'strategy-selection', 'goal-management', 'ordinal-organization', 
            'temporal-organization', 'context-alignment', 'verification', 
            'backtracking', 'conceptual-level-processing', 
            'decomposition-and-integration', 'representational-restructuring', 
            'abstraction', 'backward-chaining', 'productivity', 
            'hierarchical-organization', 'adaptive-detail-management', 
            'pattern-recognition', 'spatial-organization', 'network-organization', 
            'context-awareness'
        ]
        
        cmap_tab20 = plt.cm.get_cmap("Set2")
        cmap_tab20b = plt.cm.get_cmap("Set3")
        colors_list = [cmap_tab20(i) for i in range(cmap_tab20.N)] + \
                    [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        
        element_color_map = {}
        for i, element in enumerate(all_elements):
            r, g, b, _ = colors_list[i % len(colors_list)]
            element_color_map[element] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        
        default_color = "#AAAAAA"

        # --- 3. Compute Node "Levels" (same as before) ---
        seq_edges = [(u, v) for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == "next"]

        seq_graph = nx.DiGraph()
        seq_graph.add_nodes_from(graph.nodes())
        seq_graph.add_edges_from(seq_edges)

        if seq_graph.number_of_edges() > 0:
            try:
                levels = nx.topological_generations(seq_graph)
                level_map = {}
                for depth, layer in enumerate(levels):
                    for node in layer:
                        level_map[node] = depth
            except nx.NetworkXUnfeasible:
                level_map = {n: 0 for n in graph.nodes()}
        else:
            level_map = {n: 0 for n in graph.nodes()}

        # --- 4. Use Graphviz DOT Layout ---

        current_nodes = graph.number_of_nodes()
        title_label = f"""<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD COLSPAN="2"><B>{problem_type} (Nodes: {current_nodes} / {max_nodes})</B></TD></TR>
        <TR><TD COLSPAN="2"><HR></HR></TD></TR>
        <TR><TD ALIGN="RIGHT">Next:</TD><TD ALIGN="LEFT"><FONT COLOR="#3498db">━━━</FONT> (solid)</TD></TR>
        <TR><TD ALIGN="RIGHT">Contains:</TD><TD ALIGN="LEFT"><FONT COLOR="#e74c3c">‑ ‑ ‑</FONT> (dashed)</TD></TR>
        <TR><TD ALIGN="RIGHT">Parallel:</TD><TD ALIGN="LEFT"><FONT COLOR="#2ecc71">. . .</FONT> (dotted)</TD></TR>
        </TABLE>
        >"""

        A = pgv.AGraph(directed=True, 
                        rankdir="TB",
                        labelloc="t",
                        label=title_label,
                        fontsize="14",
                        pad="0.5",
                        splines="spline") # Use curved lines

        A.graph_attr['fontname'] = font_name
        A.node_attr['fontname'] = font_name
        A.edge_attr['fontname'] = font_name

        
        for node in graph.nodes():
            # Get the base color
            base_color = element_color_map.get(node, default_color)
            # Generate light fill and dark border/font
            fill_color, font_and_border_color = self._get_modern_colors(base_color)

            A.add_node(
                node,
                label=node,
                shape="circle",
                style="filled",
                fillcolor=fill_color,
                color=font_and_border_color,
                fontcolor=font_and_border_color,
                penwidth="2.5",
                fontsize="12"
            )

        # Add edges
        for u, v, d in graph.edges(data=True):
            e_type = d.get("edge_type", "next")
            
            color = {"next": "#3498db",
                    "contains": "#e74c3c",
                    "parallel": "#2ecc71"}[e_type]

            style = {"next": "solid",
                    "contains": "dashed",
                    "parallel": "dotted"}[e_type]

            A.add_edge(u, v,
                    color=color,
                    penwidth="2.5",
                    style=style,
                    arrowsize="1.0")

        # Enforce ranks so sequential layers align
        layers = {}
        for n, lvl in level_map.items():
            layers.setdefault(lvl, []).append(n)

        for lvl_nodes in layers.values():
            A.add_subgraph(lvl_nodes, rank="same")

        # Save layout
        A.layout(prog="dot")
        A.draw(output_file)
        print(f"Saved graph to: {output_file}")
    
    def _get_modern_colors(self, base_hex: str):
        """
        Generates a light fill color and a dark border/font color from a 
        single base hex color.
        """
        # 1. Convert hex to RGB normalized to [0, 1]
        base_hex = base_hex.lstrip('#')
        r, g, b = [int(base_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        
        # 2. Convert RGB to HLS (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # 3. Create light (fill) and dark (border/font) versions
        # Light version: increase lightness (e.g., 80% towards white)
        light_l = l + (1.0 - l) * 0.8
        # Dark version: decrease lightness (e.g., 60% towards black)
        dark_l = l * 0.6
        
        # 4. Convert back to RGB
        light_r, light_g, light_b = colorsys.hls_to_rgb(h, light_l, s)
        dark_r, dark_g, dark_b = colorsys.hls_to_rgb(h, dark_l, s)
        
        # 5. Helper to convert normalized RGB back to hex
        def to_hex(nr, ng, nb):
            return f"#{int(nr*255):02x}{int(ng*255):02x}{int(nb*255):02x}"
            
        fill_color = to_hex(light_r, light_g, light_b)
        border_color = to_hex(dark_r, dark_g, dark_b)

        if dark_r > 0.95:
            border_color = "#172849" # Dark blue
        
        return fill_color, border_color

# =============================================================================
# FILE: structure_guidance/generate_steered_traces.py
# =============================================================================

import os
import json
import asyncio
import argparse
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel
import json_repair
from openai import AsyncOpenAI
import networkx as nx
import numpy as np
import random


class ReasoningTrace(BaseModel):
    final_answer: str


class ElementGuidedReasoning:
    """Generate reasoning traces guided by problem-type-specific element graphs."""
    
    def __init__(self, args, span_tree, consensus_finder, problem_type_graphs):
        self.args = args
        self.span_tree = span_tree
        self.consensus_finder = consensus_finder
        self.problem_type_graphs = problem_type_graphs
        self.path_to_question_info = args.path_to_question_info
        self.sampled_questions = {}
    
    def sample_questions_per_type(self):
        """Sample questions for each problem type, balancing successful/unsuccessful traces."""
        print("\nSampling questions per problem type...")

        with open(self.args.path_to_question_info, 'r') as f:
            eval_json = json.load(f)
        all_question_info = {str(question_info['question_id']): question_info['question'] for question_info in eval_json if question_info['model'] == 'Qwen3-8B'}
        
        for problem_type in self.problem_type_graphs.keys():
            # Get all problems of this type
            problems_of_type = [
                (pid, problem) for pid, problem in self.span_tree.problems.items()
                if (problem.problem_type == problem_type) and (problem.problem_id in all_question_info)
            ]
            
            # Separate by success/failure for the target model
            successful = []
            unsuccessful = []
            
            for pid, problem in problems_of_type:
                if self.args.model_name in problem.correctness:
                    if problem.correctness[self.args.model_name]:
                        successful.append(pid)
                    else:
                        unsuccessful.append(pid)
            
            print(f"\nProblem type: {problem_type}")
            print(f"  Total problems: {len(problems_of_type)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Unsuccessful: {len(unsuccessful)}")
            
            # Sample with desired distribution
            n_samples = min(self.args.samples_per_type, len(problems_of_type))
            target_successful = n_samples // 2
            target_unsuccessful = n_samples - target_successful
            
            # Adjust if we don't have enough of one type
            actual_successful = min(target_successful, len(successful))
            actual_unsuccessful = min(target_unsuccessful, len(unsuccessful))
            
            # If one category is short, try to take more from the other
            if actual_successful < target_successful:
                actual_unsuccessful = min(n_samples - actual_successful, len(unsuccessful))
            if actual_unsuccessful < target_unsuccessful:
                actual_successful = min(n_samples - actual_unsuccessful, len(successful))
            
            # Sample
            sampled_successful = random.sample(successful, actual_successful) if successful else []
            sampled_unsuccessful = random.sample(unsuccessful, actual_unsuccessful) if unsuccessful else []
            
            sampled = sampled_successful + sampled_unsuccessful

            question_text = {id:all_question_info[id] for id in sampled}
            
            print(f"  Sampled: {len(sampled)} ({len(sampled_successful)} successful, {len(sampled_unsuccessful)} unsuccessful)")
            
            self.sampled_questions[problem_type] = {
                'problem_ids': sampled,
                'problem_texts': question_text,
                'successful_ids': sampled_successful,
                'unsuccessful_ids': sampled_unsuccessful
            }
    
    def graph_to_prompt(self, graph: nx.DiGraph) -> str:
        """Convert a element graph to a natural language prompt."""
        if graph.number_of_nodes() == 0:
            return "No specific reasoning structure required."
        
        prompt_parts = [
            "Follow this reasoning structure when solving the problem:",
            ""
        ]
        
        # Group elements by their connections
        nodes_info = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            support = node_data.get('support', 0)
            nodes_info.append((node, support))
        
        # Sort by support (most important first)
        nodes_info.sort(key=lambda x: x[1], reverse=True)
        
        # Add nodes with their relationships
        prompt_parts.append("Key reasoning elements to include:")
        for i, (node, support) in enumerate(nodes_info[:10], 1):  # Limit to top 10
            prompt_parts.append(f"{i}. {node}")
        
        # Add edge information for structure
        if graph.number_of_edges() > 0:
            prompt_parts.append("\nReasoning flow:")
            
            # Get edges sorted by support
            edges_info = []
            for u, v, data in graph.edges(data=True):
                edge_type = data.get('edge_type', 'next')
                support = data.get('support', 0)
                edges_info.append((u, v, edge_type, support))
            
            edges_info.sort(key=lambda x: x[3], reverse=True)
            
            for u, v, edge_type, support in edges_info[:15]:  # Limit to top 15
                if edge_type == 'next':
                    prompt_parts.append(f"  - {u} → {v}")
                elif edge_type == 'contains':
                    prompt_parts.append(f"  - {u} (contains) {v}")
                elif edge_type == 'parallel':
                    prompt_parts.append(f"  - {u} (parallel with) {v}")
        
        return "\n".join(prompt_parts)
    
    async def generate_guided_trace(self, question_id, question_info, graph, semaphore, condition_type):
        """Generate a reasoning trace guided by the element graph."""
        async with semaphore:
            # Build the prompt with element guidance
            element_guidance = self.graph_to_prompt(graph)
            
            base_question = question_info['question']
            
            if condition_type == 'guided':
                prompt = f"""{element_guidance}

Question:
{base_question}

Reason through this question following the structure above, then provide your final answer in the following JSON format:
{{
    "final_answer": str
}}
"""
            else:  # baseline - no guidance
                prompt = f"""Reason then answer the following question to the best of your abilities.
Question:
{base_question}

Format your answer in the following JSON format:
{{
    "final_answer": str
}}
"""
            
            try:
                if self.args.no_parser:
                    chat_response = await self.args.client.chat.completions.create(
                        model=self.args.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        top_p=0.95,
                        extra_body={
                            "top_k": 20,
                            "min_p": 0,
                            "n": 1,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                    
                    # Parse response
                    output = chat_response.choices[0]
                    
                    # Try to extract reasoning and answer
                    content = output.message.content
                    
                    if "```json" in content:
                        reasoning_content = content.split("```json")[0].strip()
                        try:
                            final_answer = json_repair.loads(content.split("```json")[-1].strip())['final_answer']
                        except:
                            final_answer = content.split("```json")[-1].strip()
                    elif '"final_answer":' in content:
                        reasoning_content = content.split('"final_answer":')[0].strip()
                        try:
                            final_answer = json_repair.loads('{' + content.split('"final_answer":')[1].strip())['final_answer']
                        except:
                            final_answer = content.split('"final_answer":')[1].strip()
                    else:
                        reasoning_content = content
                        final_answer = content
                    
                    if len(reasoning_content) < 3:
                        reasoning_content = ""
                    
                else:
                    chat_response = await self.args.client.chat.completions.create(
                        model=self.args.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "answer",
                                "schema": ReasoningTrace.model_json_schema()
                            },
                        },
                        max_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        top_p=0.95,
                        extra_body={
                            "top_k": 20,
                            "min_p": 0,
                            "n": 1,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                    
                    output = chat_response.choices[0]
                    reasoning_content = output.message.reasoning_content if hasattr(output.message, 'reasoning_content') else ""
                    final_answer = json_repair.loads(output.message.content)['final_answer']
                
                result = {
                    "reasoning": reasoning_content,
                    "answer": final_answer,
                    "condition": condition_type,
                    "input_tokens": chat_response.usage.prompt_tokens,
                    "output_tokens": chat_response.usage.completion_tokens
                }
                
                return question_id, result
                
            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                return question_id, None
    
    async def run_guided_generation(self):
        """Run guided generation for all sampled questions."""
        print("\nGenerating guided reasoning traces...")
        
        semaphore = asyncio.Semaphore(self.args.max_concurrent)
        
        results = {}
        
        for problem_type, sample_info in self.sampled_questions.items():
            print(f"\nProcessing problem type: {problem_type}")
            graph = self.problem_type_graphs[problem_type]
            
            results[problem_type] = {
                'questions': {},
                'graph_info': {
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges()
                }
            }
            
            tasks = []
            for question_id in sample_info['problem_ids']:
                question_info = self.span_tree.problems[question_id]
                
                # Get original question data
                orig_question = {
                    'question': question_info.problem_id,  # You might need to load actual question text
                    'reference_answer': 'N/A',
                    'task': question_info.task_category
                }
                
                # Generate both guided and baseline versions
                tasks.append(
                    self.generate_guided_trace(
                        question_id, orig_question, graph, semaphore, 'guided'
                    )
                )
                tasks.append(
                    self.generate_guided_trace(
                        question_id, orig_question, graph, semaphore, 'baseline'
                    )
                )
            
            # Process with progress bar
            completed_results = []
            for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {problem_type}"):
                result = await coro
                completed_results.append(result)
            
            # Organize results
            for question_id, result in completed_results:
                if result:
                    if question_id not in results[problem_type]['questions']:
                        results[problem_type]['questions'][question_id] = {
                            'problem_id': question_id,
                            'original_correctness': self.span_tree.problems[question_id].correctness.get(self.args.model_name, None),
                            'guided': None,
                            'baseline': None
                        }
                    
                    if result['condition'] == 'guided':
                        results[problem_type]['questions'][question_id]['guided'] = result
                    else:
                        results[problem_type]['questions'][question_id]['baseline'] = result
        
        return results
    
    async def run(self):
        """Main execution flow."""
        # Load data
        self.load_span_tree()
        
        # Build graphs
        self.build_problem_type_graphs()
        
        # Sample questions
        self.sample_questions_per_type()
        
        # Generate guided traces
        results = await self.run_guided_generation()
        
        # Save results
        output_path = os.path.join(
            self.args.output_dir,
            f"{self.args.model_name.split('/')[-1]}_guided_reasoning.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Print summary statistics
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print summary statistics."""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        for problem_type, data in results.items():
            print(f"\nProblem Type: {problem_type}")
            print(f"  Graph: {data['graph_info']['num_nodes']} nodes, {data['graph_info']['num_edges']} edges")
            print(f"  Questions processed: {len(data['questions'])}")
            
            guided_complete = sum(1 for q in data['questions'].values() if q['guided'] is not None)
            baseline_complete = sum(1 for q in data['questions'].values() if q['baseline'] is not None)
            
            print(f"  Guided completions: {guided_complete}")
            print(f"  Baseline completions: {baseline_complete}")


async def main():
    parser = argparse.ArgumentParser(description="Generate element-guided reasoning traces")
    
    # Model and API settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=25000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_concurrent", type=int, default=150)
    parser.add_argument("--no_parser", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    
    # Sampling parameters
    parser.add_argument("--samples_per_type", type=int, default=50,
                       help="Number of samples per problem type")
    
    args = parser.parse_args()
    
    # Setup API client
    args.client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
        timeout=2400
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run the pipeline
    pipeline = ElementGuidedReasoning(args)
    await pipeline.run()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    asyncio.run(main())

# =============================================================================
# TEXT FILES - ANNOTATION PROMPTS AND TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# FILE: element_annotation/system_prompt.txt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_TXT = r"""You are an expert cognitive scientist conducting research on reasoning behaviors. Analyze reasoning traces to identify specific text spans that demonstrate cognitive capabilities.

## Task
Analyze the provided reasoning trace and output JSON with:
1. **"explanation"**: One-sentence assessment of capability presence
2. **"span_analysis"**: Your reasoning for identifying relevant text segments  
3. **"spans"**: Character indices [[start, end], [start, end]] for spans that demonstrate the capability
4. **"score"**: 0 (absent), 1 (partial), or 2 (present)

## Span Guidelines
- **Character indices**: 0-based counting, reasoning_trace[start:end] must extract complete, meaningful text
- **Relevance**: Only include text that clearly demonstrates the target capability
- **Completeness**: Use complete sentences or phrases that make sense alone
- **Precision**: Prefer shorter, focused spans over longer, unfocused ones

## Edge Cases
- No clear capability → spans: []
- Uncertain boundaries → use shorter, precise spans

## Quality Check
Verify: accurate indices, clear capability demonstration, logical span_analysis, aligned score.
"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/template_prompt.txt
# -----------------------------------------------------------------------------
TEMPLATE_PROMPT_TXT = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: {p_type}
Here is the optimal {node_num}-node behavior structure:

{structure_info}

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: element_annotation/abstraction-trace-prompt.txt
# -----------------------------------------------------------------------------
ABSTRACTION_PROMPT = r"""# Annotation Guidelines: Abstraction in the Reasoning Process with Span Identification

## Definition
**Abstraction** is the ability to extract general principles from specific instances. In reasoning traces, abstraction refers to when the participant demonstrates the ability to identify underlying concepts, generalize from concrete examples, derive broader principles, and apply general concepts across different contexts.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates abstraction:

1. **Generalization from examples**: Does the participant derive general principles from specific instances?
   - Look for extraction of broader patterns or rules from concrete cases
   - Check if the participant identifies commonalities that transcend specific examples

2. **Concept formation**: Does the participant form abstract concepts beyond surface features?
   - Look for formulation of higher-level constructs or categories
   - Check if the participant develops conceptual frameworks that organize specific instances

3. **Level shifting**: Does the participant move between concrete and abstract levels?
   - Look for transitions between specific examples and general principles
   - Check if the participant can apply abstract ideas to specific cases and extract abstractions from specifics

4. **Cross-domain application**: Does the participant apply principles across different domains?
   - Look for transfer of abstract concepts between distinct contexts
   - Check if the participant recognizes when the same abstract principle applies in different situations

## Label Levels

**0 - Absent**: The reasoning trace shows little to no abstraction. The participant focuses on specific details or concrete examples without extracting general principles or forming abstract concepts.

**1 - Partially Present**: The reasoning trace shows some abstraction, but with limited depth or inconsistent application. The participant occasionally generalizes from examples or forms basic abstractions, but doesn't consistently operate at an abstract level or effectively move between concrete and abstract.

**2 - Present**: The reasoning trace shows clear abstraction throughout. The participant consistently generalizes from specific instances, forms sophisticated abstract concepts, effectively moves between concrete and abstract levels, and applies principles across different domains.

## Span Identification Instructions

In addition to scoring the overall presence of abstraction, you must identify specific spans (text segments) in the reasoning trace that demonstrate abstraction. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of abstraction (generalization from examples, concept formation, level shifting, or cross-domain application)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to abstraction
- **Multiple spans**: Identify all significant instances of abstraction, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of abstraction are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates abstraction

### How to Identify Valid Spans
1. **Look for abstraction language**: Words like "general," "common," "pattern," "principle," "abstract," "applies across," "transcends"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant moving from specific examples to general principles or concepts?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether abstraction is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate abstraction. Describe what specific phrases or sentences show generalization, concept formation, level shifting, or cross-domain application, and explain why they demonstrate abstraction. This guides you to identify the character indices. Use an empty string "" if no abstraction is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate abstraction. Use an empty list [] if no spans demonstrate abstraction.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To calculate the area of this rectangle, I'll multiply the length by the width. The rectangle has a length of 8 cm and a width of 5 cm. Therefore, Area = Length × Width = 8 cm × 5 cm = 40 cm². The area of the rectangle is 40 square centimeters."

Annotation:
```json
{{
  "explanation": "The participant performs a straightforward calculation on a specific example without any generalization, concept formation, level shifting between concrete and abstract, or application of principles beyond this single case.",
  "span_analysis": "no signs of abstraction at all.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Looking at these three historical revolutions—the American Revolution, the French Revolution, and the Russian Revolution—I can identify some common elements despite their different contexts. All three involved issues of economic inequality, though the specific grievances varied: taxation without representation in America, feudal privileges in France, and industrial working conditions in Russia. Each revolution also included a radicalizing period where initial moderate reforms gave way to more extreme measures. The outcomes differed significantly, with America establishing a stable democracy relatively quickly, France experiencing the Reign of Terror followed by Napoleon's rise, and Russia transitioning to a communist state under Lenin and later Stalin. These historical examples show how revolutions, while sharing some characteristics, can follow different trajectories depending on their specific circumstances."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some abstraction by identifying common elements across different historical revolutions, but the analysis remains largely descriptive and tied to specific historical details without developing sophisticated abstract frameworks or applying principles beyond the given examples.",
  "span_analysis": "Looking for abstraction in this text: The opening sentence 'I can identify some common elements despite their different contexts' shows basic generalization from specific examples. The final sentence 'These historical examples show how revolutions...can follow different trajectories depending on their specific circumstances' makes a general observation about revolutions as a category. However, most of the text focuses on specific historical details rather than abstract principles.",
  "spans": [[0, 190], [763, 923]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "Examining these diverse conflict resolution scenarios—from international treaties to corporate mergers to family therapy—I can identify foundational principles that transcend their surface differences. At the most abstract level, all successful conflict resolution processes share a transformation pattern: they convert zero-sum framing (where one party's gain is another's loss) into positive-sum framing (where mutual benefit becomes possible). This principle manifests differently across domains but represents the same fundamental shift in conceptualization. From these specific instances, I can extract a general principle I'll call 'interest-based reconciliation.' This abstract principle involves distinguishing between positions (what parties say they want) and interests (why they want it). In the corporate merger, Company A's position was demanding full brand retention, but their underlying interest was market recognition. In the family therapy case, the teenager's position was demanding complete autonomy, but the underlying interest was respect and acknowledgment of growing maturity. This principle operates at three levels of abstraction: concrete level (specific demands and concessions in each case), mid-level abstraction (domain-specific applications like negotiation tactics, therapy techniques), and high-level abstraction (universal principles of human motivation and conflict dynamics). Moving between these levels, we can see how the same abstract principle manifests in different contexts. For example, the concept of 'saving face' appears critically important across these diverse scenarios—from nations needing to maintain dignity during treaty negotiations to family members needing to preserve self-image during reconciliation. I can apply these abstracted principles predictively: in a novel scenario like an environmental dispute between developers and conservation groups, we would expect successful resolution to involve reframing from zero-sum to positive-sum, identifying interests behind positions, and creating face-saving mechanisms for all parties. At the highest level of abstraction, these cases reveal a meta-principle about conflict itself: sustainable resolution requires addressing both objective factors (resources, rights, power) and subjective factors (perception, identity, meaning). This principle transcends specific domains and provides a conceptual framework for understanding conflict resolution universally."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated abstraction by extracting general principles from diverse conflict resolution examples, forming explicit abstract concepts like 'interest-based reconciliation,' systematically moving between concrete examples and increasingly abstract levels, developing a conceptual framework that organizes the domain, and applying these abstract principles predictively to novel scenarios.",
  "span_analysis": "Analyzing this text for abstraction: The opening 'I can identify foundational principles that transcend their surface differences' shows cross-domain application across international, corporate, and family contexts. The phrase 'At the most abstract level, all successful conflict resolution processes share a transformation pattern' demonstrates concept formation and level shifting. The text 'From these specific instances, I can extract a general principle I'll call interest-based reconciliation' explicitly shows concept formation by creating a named abstract principle. The section about 'three levels of abstraction: concrete level...mid-level...high-level' demonstrates sophisticated level shifting and meta-abstraction. The phrase 'I can apply these abstracted principles predictively' to environmental disputes shows cross-domain application to novel scenarios. Finally, 'At the highest level of abstraction, these cases reveal a meta-principle' demonstrates the most sophisticated concept formation. These sections show advanced abstraction through concept formation, level shifting, and cross-domain application.",
  "spans": [[0, 201], [202, 446], [563, 670], [1101, 1412], [1760, 2090], [2091, 2335]],
  "score": 2
}}
```

## Important Notes for Annotators

1. **Character Counting**: Carefully count character indices to ensure spans can be extracted correctly using reasoning_trace[start_index:end_index]
2. **Span Boundaries**: Ensure spans capture complete thoughts or sentences for meaningful analysis
3. **Span Overlap**: Spans may overlap if a text segment demonstrates multiple types of abstraction
4. **Context Sensitivity**: Consider the full context when determining if a span truly demonstrates abstraction
5. **Quality over Quantity**: Focus on identifying the most clear and significant instances of abstraction rather than marking every possible span
6. **Consistency**: Use consistent criteria across all annotations for reliable span identification
7. **Validation**: Always include the exact text of each span to allow for verification of character indices

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of abstraction is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:
"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/adaptive-detail-management-trace-prompt.txt
# -----------------------------------------------------------------------------
ADAPTIVE_DETAIL_MANAGEMENT_PROMPT = r"""# Annotation Guidelines: Adaptive Detail Management in the Reasoning Process

## Definition
**Adaptive detail management** is the ability to adjust the level of detail based on reasoning requirements. In reasoning traces, adaptive detail management refers to when the participant demonstrates the ability to shift between different levels of abstraction, expand or compress information as needed, determine appropriate levels of specificity for different aspects of reasoning, and dynamically manage detail levels throughout the reasoning process.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates adaptive detail management:

1. **Abstraction level shifting**: Does the participant move between different levels of abstraction?
   - Look for transitions between high-level principles and specific details
   - Check if the participant can zoom out for broad perspectives and zoom in for fine details

2. **Detail calibration**: Does the participant provide appropriate levels of detail for different aspects?
   - Look for more detailed treatment of critical issues and less for peripheral ones
   - Check if the participant determines suitable granularity based on importance

3. **Information expansion/compression**: Does the participant expand or compress information as needed?
   - Look for elaboration when detail is necessary and summarization when it isn't
   - Check if the participant can unpack concepts or condense information adaptively

4. **Dynamic granularity**: Does the participant adjust detail levels throughout the reasoning process?
   - Look for changes in specificity as reasoning progresses
   - Check if the participant increases or decreases detail in response to emerging needs

## Label Levels

**0 - Absent**: The reasoning trace shows little to no adaptive detail management. The participant maintains a uniform level of detail regardless of importance or uses inappropriate levels of specificity throughout.

**1 - Partially Present**: The reasoning trace shows some adaptive detail management, but with limited flexibility or inconsistent application. The participant occasionally adjusts detail levels but may not consistently provide appropriate granularity or shift between abstraction levels effectively.

**2 - Present**: The reasoning trace shows clear adaptive detail management throughout. The participant consistently shifts between abstraction levels, calibrates detail appropriately, expands or compresses information effectively, and dynamically adjusts granularity in response to reasoning requirements.

## Span Identification Instructions

In addition to scoring the overall presence of adaptive detail management, you must identify specific spans (text segments) in the reasoning trace that demonstrate adaptive detail management. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of adaptive detail management (abstraction level shifting, detail calibration, information expansion/compression, or dynamic granularity)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to adaptive detail management
- **Multiple spans**: Identify all significant instances of adaptive detail management, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of adaptive detail management are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates adaptive detail management

### How to Identify Valid Spans
1. **Look for detail management language**: Words like "more detail," "specifically," "broadly," "zoom in," "zoom out," "elaborate," "summarize," "at a high level," "getting specific"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant adjusting detail levels or managing information granularity?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether adaptive detail management is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate adaptive detail management. Describe what specific phrases or sentences show abstraction level shifting, detail calibration, information expansion/compression, or dynamic granularity, and explain why they demonstrate adaptive detail management. This guides you to identify the character indices. Use an empty string "" if no adaptive detail management is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate adaptive detail management. Use an empty list [] if no spans demonstrate adaptive detail management.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To calculate the compound interest, we use the formula A = P(1+r)^t, where A is the final amount, P is the principal, r is the interest rate, and t is the time in years. For this problem, P = $1000, r = 5% = 0.05, and t = 3 years. Substituting these values into the formula: A = $1000(1+0.05)^3 = $1000(1.05)^3 = $1000(1.157625) = $1157.63. Therefore, the final amount after 3 years is $1157.63, and the compound interest earned is $1157.63 - $1000 = $157.63."

Annotation:
```json
{{
  "explanation": "The participant maintains a uniform level of detail throughout its reasoning, simply working through a formula application without adjusting abstraction levels, calibrating detail based on importance, or dynamically managing information density.",
  "span_analysis": "This text shows no adaptive detail management - it maintains a consistent level of detail throughout without any abstraction level shifting, detail calibration, information expansion/compression, or dynamic granularity adjustments.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Let me analyze the impact of the new policy on different stakeholders. For consumers, the policy will likely result in moderately higher prices in the short term as companies pass on compliance costs. Looking at specific numbers, research suggests a 2-3% price increase across affected products, which translates to approximately $50-75 annually for the average household.

For businesses, the impact varies significantly by size. Large corporations with existing compliance systems can adapt relatively easily—implementation costs represent less than 0.1% of annual revenue for major players in the industry. For small businesses, however, I should examine the impacts more carefully. These entities face disproportionate compliance burdens, with costs potentially reaching 2-5% of revenue. The specific requirements include new reporting systems, staff training, and potential reformulation of products, each with its own timeline and expense.

At a broader economic level, the policy aims to address market externalities that previously cost the public sector approximately $2.3 billion annually. The projected net social benefit is positive, though the distribution of costs and benefits deserves consideration."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some adaptive detail management by providing more specific numerical detail for consumer and small business impacts while keeping broader economic impacts more general, but it doesn't consistently shift between abstraction levels or dynamically adjust detail throughout the reasoning process.",
  "span_analysis": "Looking for adaptive detail management in this text: The phrase 'Looking at specific numbers, research suggests a 2-3% price increase across affected products, which translates to approximately $50-75 annually for the average household' shows information expansion by providing specific numerical details. The text 'I should examine the impacts more carefully' demonstrates dynamic granularity by explicitly adjusting detail level for small businesses. The phrase 'At a broader economic level' shows abstraction level shifting by moving to a higher level of analysis. However, the adjustments are limited and not consistently applied throughout.",
  "spans": [[201, 372], [641, 684], [947, 974]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To analyze the novel's significance, I'll need to manage detail adaptively across multiple levels of literary analysis.

At the highest level of abstraction, this work represents a pivotal transition between Victorian and modernist sensibilities in English literature. This contextual frame is sufficient to establish its historical position without needing extensive period detail.

Moving to a more specific level, let me examine three key innovations that define its importance: narrative technique, thematic concerns, and linguistic experimentation. For narrative technique, a moderate level of detail is appropriate. The novel introduced free indirect discourse as its primary mode, allowing for unprecedented psychological realism. I won't detail every instance of this technique—a brief illustration from Chapter 3, where the protagonist's thoughts blend seamlessly with narrative description, suffices to demonstrate the method.

For thematic analysis, greater elaboration is warranted since this represents the novel's most influential contribution. I'll expand considerably here: the work's exploration of class mobility introduced a complex framework involving four distinct social strata rather than the binary divisions common in earlier works. Specifically, the protagonist navigates between: 1) the traditional aristocracy, 2) the newly wealthy industrial class, 3) the educated but financially precarious middle class, and 4) the urban working class. Each boundary crossing reveals different social mechanisms, which I'll analyze with detailed examples from pivotal scenes in Chapters 2, 7, and 12.

Regarding linguistic experimentation, I can be more concise as this aspect, while innovative, had less lasting influence. The author's occasional use of regional dialects and technical terminology was noteworthy but not revolutionary.

As I move into discussing critical reception, I'll shift back to a higher level of abstraction, summarizing the broad patterns of response rather than cataloging individual critics' views. A brief mention of the Wilson review (1922) provides sufficient specific grounding.

Finally, for contemporary relevance, I'll increase detail again to examine how the novel's treatment of technological anxiety specifically prefigured current debates about automation and social displacement. The factory scene in Chapter 9 deserves particular attention, so I'll zoom in to analyze the symbolic elements and psychological portrayals that resonate with current concerns."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated adaptive detail management by explicitly navigating between different abstraction levels, strategically allocating detail based on importance, deliberately expanding information for critical elements while compressing others, and dynamically adjusting specificity throughout the analysis as reasoning needs change.",
  "span_analysis": "Analyzing this text for adaptive detail management: The opening 'I'll need to manage detail adaptively across multiple levels of literary analysis' explicitly establishes adaptive detail management as the approach. The phrase 'At the highest level of abstraction' demonstrates abstraction level shifting. The text 'Moving to a more specific level' shows dynamic granularity by explicitly changing detail levels. The phrase 'For narrative technique, a moderate level of detail is appropriate' demonstrates detail calibration based on importance. The text 'For thematic analysis, greater elaboration is warranted' shows information expansion for critical elements. The phrase 'I can be more concise as this aspect, while innovative, had less lasting influence' demonstrates information compression for less important elements. These show sophisticated adaptive detail management across all dimensions.",
  "spans": [[37, 119], [121, 268], [384, 415], [554, 621], [938, 1058], [1654, 1737]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of adaptive detail management is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/backtracking-trace-prompt.txt
# -----------------------------------------------------------------------------
BACKTRACKING_PROMPT = r"""# Annotation Guidelines: Backtracking in the Reasoning Process

## Definition
**Backtracking** is the ability to identify unproductive paths and return to earlier decision points. In reasoning traces, backtracking refers to when the participant demonstrates the ability to recognize when a line of reasoning is not yielding progress, abandon that path, return to a previous state, and explore alternative directions.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates backtracking:

1. **Dead-end recognition**: Does the participant identify when a reasoning path is unproductive?
   - Look for acknowledgment that a current approach isn't working
   - Check if the participant recognizes when it has reached an impasse

2. **Path abandonment**: Does the participant explicitly abandon unfruitful approaches?
   - Look for decisions to stop pursuing unproductive lines of reasoning
   - Check if the participant willingly gives up on approaches that aren't yielding results

3. **Return to decision points**: Does the participant go back to earlier points in the reasoning?
   - Look for returns to previous states or junctures
   - Check if the participant identifies where to backtrack to

4. **Alternative exploration**: Does the participant try different approaches after backtracking?
   - Look for exploration of multiple solution paths
   - Check if the participant attempts new strategies after abandoning unsuccessful ones

## Label Levels

**0 - Absent**: The reasoning trace shows little to no backtracking. The participant persists with initial approaches without recognizing dead ends, abandoning unproductive paths, or exploring alternatives.

**1 - Partially Present**: The reasoning trace shows some backtracking, but with limited systematicity or incomplete application. The participant occasionally recognizes unproductive paths and tries alternatives, but may not consistently or effectively backtrack when appropriate.

**2 - Present**: The reasoning trace shows clear backtracking throughout. The participant consistently identifies unproductive paths, explicitly abandons them, returns to appropriate decision points, and explores alternative approaches.

## Span Identification Instructions

In addition to scoring the overall presence of backtracking, you must identify specific spans (text segments) in the reasoning trace that demonstrate backtracking. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of backtracking (dead-end recognition, path abandonment, return to decision points, or alternative exploration)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to backtracking
- **Multiple spans**: Identify all significant instances of backtracking, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of backtracking are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates backtracking

### How to Identify Valid Spans
1. **Look for backtracking language**: Words like "this isn't working," "let me try a different approach," "backtrack," "abandon," "switch to," "reconsider," "try again"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant recognizing dead ends, abandoning paths, or exploring alternatives?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether backtracking is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate backtracking. Describe what specific phrases or sentences show dead-end recognition, path abandonment, return to decision points, or alternative exploration, and explain why they demonstrate backtracking. This guides you to identify the character indices. Use an empty string "" if no backtracking is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate backtracking. Use an empty list [] if no spans demonstrate backtracking.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the area of the triangle with vertices at (0,0), (3,0), and (0,4), I'll use the formula Area = (1/2) × base × height. The base of the triangle is along the x-axis from (0,0) to (3,0), so the base length is 3 units. The height is the perpendicular distance from the third vertex to the base, which is 4 units. Applying the formula: Area = (1/2) × 3 × 4 = 6 square units."

Annotation:
```json
{{
  "explanation": "The participant applies a straightforward calculation without encountering any obstacles or unproductive paths, so there is no opportunity or need for backtracking in this simple reasoning task.",
  "span_analysis": "This text shows no backtracking - it follows a single, direct path to the solution without any dead-end recognition, path abandonment, or alternative exploration.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this quadratic equation, 2x² - 5x - 3 = 0, I'll try factoring first. I need to find two numbers that multiply to -6 and add to -5. Let me try some pairs: (-6,1), (-3,2), etc. Hmm, this isn't working out easily.

Let me try a different approach. I'll use the quadratic formula: x = (-b ± √(b² - 4ac))/2a, where a = 2, b = -5, and c = -3. Substituting these values:
x = (5 ± √(25 + 24))/4
x = (5 ± √49)/4
x = (5 ± 7)/4

So x = (5 + 7)/4 = 3 or x = (5 - 7)/4 = -1/2. Therefore, the solutions are x = 3 and x = -1/2."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some backtracking by recognizing that the factoring approach isn't working easily, abandoning that path, and switching to the quadratic formula, but this represents a relatively simple case of backtracking without multiple attempts or complex exploration of alternatives.",
  "span_analysis": "Looking for backtracking in this text: The phrase 'Hmm, this isn't working out easily' shows dead-end recognition by acknowledging the current approach is unproductive. The text 'Let me try a different approach' demonstrates path abandonment and alternative exploration by explicitly switching strategies. These show basic backtracking but without complex multiple attempts.",
  "spans": [[184, 219], [221, 253]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To prove this theorem about the properties of this mathematical structure, I'll try using induction.

First, I need to establish the base case for n=1. For n=1, the left side equals 2¹-1, while the right side equals 1, so the base case holds.

Now for the inductive step, I'll assume the statement is true for n=k and try to prove it for n=k+1. Starting with the left side for n=k+1...

Hmm, this approach is getting quite complicated. The algebraic manipulation is becoming unwieldy, and I'm not seeing a clear path to connect to the right side of the equation. Let me backtrack and try a different proof strategy.

Instead of induction, let's try a direct algebraic approach. I'll manipulate the expression...

Actually, this is also becoming overly complex. The algebraic transformations don't seem to be simplifying in a helpful way. Let me abandon this path as well and reconsider the problem from a different angle.

Looking at the theorem statement again, I notice it has a structure reminiscent of a combinatorial identity. Let me try a combinatorial proof approach.

The left side can be interpreted as counting the number of ways to select subsets of specific sizes from a set of n elements. The right side represents...

Wait, I see that this interpretation doesn't quite fit either. Let me backtrack once more.

Let's return to the original expression and try yet another approach: a generating function method. If I define the generating function G(x) = ...

Now I'm making progress! By manipulating this generating function and examining the coefficient of x^n, I can show that...

Perfect! This approach successfully proves the theorem. The generating function method allowed me to establish the identity by connecting it to well-known properties of power series expansions, avoiding the algebraic complexity I encountered in my earlier attempts."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated backtracking by repeatedly recognizing when approaches are unproductive, explicitly abandoning each unsuccessful path, returning to earlier decision points to reconsider the problem from scratch, and systematically exploring multiple alternative strategies until finding one that works.",
  "span_analysis": "Analyzing this text for backtracking: The phrase 'Hmm, this approach is getting quite complicated' shows dead-end recognition. The text 'Let me backtrack and try a different proof strategy' explicitly demonstrates backtracking and alternative exploration. The phrase 'Actually, this is also becoming overly complex' shows another dead-end recognition. The text 'Let me abandon this path as well and reconsider the problem from a different angle' demonstrates path abandonment and return to decision points. The phrase 'Let me backtrack once more' shows repeated backtracking. These demonstrate sophisticated, systematic backtracking across multiple failed approaches.",
  "spans": [[387, 435], [563, 615], [713, 760], [838, 921], [1232, 1322]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of backtracking is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/backward-chaining-trace-prompt.txt
# -----------------------------------------------------------------------------
BACKWARD_CHAINING_PROMPT = r"""# Annotation Guidelines: Backward Chaining in the Reasoning Process

## Definition
**Backward chaining** is the ability to start with goals and work backward to identify prerequisites. In reasoning traces, backward chaining refers to when the participant demonstrates the ability to begin with a desired outcome, determine what would be needed to achieve it, and recursively identify the conditions that would satisfy those requirements.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates backward chaining:

1. **Goal-directed reasoning**: Does the participant start with the goal or desired outcome?
   - Look for beginning with what needs to be proved or achieved
   - Check if the participant identifies the target state before determining how to get there

2. **Prerequisite identification**: Does the participant determine what conditions are needed to achieve the goal?
   - Look for identification of necessary preconditions
   - Check if the participant works out what must be true for the goal to be achieved

3. **Recursive backtracking**: Does the participant recursively identify prerequisites of prerequisites?
   - Look for multiple levels of backward reasoning
   - Check if the participant traces chains of requirements back to initial conditions

4. **Backward direction**: Does the participant work from effects to causes or conclusions to premises?
   - Look for reasoning that moves from consequences to antecedents
   - Check if the participant works backward from goals to find starting points

## Label Levels

**0 - Absent**: The reasoning trace shows little to no backward chaining. The participant doesn't start with goals and work backward, instead using other approaches like forward reasoning from premises or lateral thinking.

**1 - Partially Present**: The reasoning trace shows some backward chaining, but with limited systematicity or incomplete application. The participant sometimes works backward from goals, but may not consistently apply recursive backtracking or may mix backward chaining with other approaches.

**2 - Present**: The reasoning trace shows clear backward chaining throughout. The participant consistently begins with goals, identifies necessary prerequisites, recursively traces prerequisites of prerequisites, and works in a backward direction from goals to initial conditions.

## Span Identification Instructions

In addition to scoring the overall presence of backward chaining, you must identify specific spans (text segments) in the reasoning trace that demonstrate backward chaining. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of backward chaining (goal-directed reasoning, prerequisite identification, recursive backtracking, or backward direction)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to backward chaining
- **Multiple spans**: Identify all significant instances of backward chaining, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of backward chaining are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates backward chaining

### How to Identify Valid Spans
1. **Look for backward chaining language**: Words like "to achieve this, I need," "working backward," "what would be required," "in order to," "this depends on," "prerequisite," "necessary condition"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant working backward from goals to identify prerequisites?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether backward chaining is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate backward chaining. Describe what specific phrases or sentences show goal-directed reasoning, prerequisite identification, recursive backtracking, or backward direction, and explain why they demonstrate backward chaining. This guides you to identify the character indices. Use an empty string "" if no backward chaining is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate backward chaining. Use an empty list [] if no spans demonstrate backward chaining.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the value of x in the equation 3x + 7 = 22, I'll solve step by step. First, I have 3x + 7 = 22. To isolate the variable, I'll subtract 7 from both sides: 3x + 7 - 7 = 22 - 7, which gives 3x = 15. Next, I'll divide both sides by 3: 3x ÷ 3 = 15 ÷ 3, resulting in x = 5. Therefore, the value of x is 5. I can verify this answer by substituting it back into the original equation: 3(5) + 7 = 15 + 7 = 22, which confirms that x = 5 is correct."

Annotation:
```json
{{
  "explanation": "The participant uses a forward approach by starting with the given equation and applying algebraic operations sequentially until reaching the solution, showing no evidence of backward reasoning from the goal to prerequisites.",
  "span_analysis": "This text shows no backward chaining - it follows a forward, sequential approach from the given equation to the solution without working backward from goals to identify prerequisites.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To determine how to increase user engagement on our app by 20% next quarter, I'll start by defining the goal. We need to reach a 20% increase in our key engagement metric, which is currently average session duration of 3.5 minutes per user.

So the goal is to reach about 4.2 minutes average session duration. What would cause users to spend more time in the app? They would need to find more valuable content and have a smoother user experience. What creates those conditions? We would need some combination of improved content recommendation algorithms, new content features, and UI improvements.

For better recommendations, we would need to improve our machine learning models, which requires better user data collection and analysis. For new content features, we would need to identify what users want most, which requires market research and competitor analysis. For UI improvements, we would need usability testing and design improvements.

Given our current resources and timeline, I recommend focusing on improving the recommendation algorithm, which has shown the highest impact on session duration in past tests."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some backward chaining by starting with the engagement goal and working backward to identify prerequisites (better content, smoother UX) and then further prerequisites for those conditions (improved algorithms, market research, usability testing), but the recursion is limited to a few levels and the approach is blended with other reasoning methods.",
  "span_analysis": "Looking for backward chaining in this text: The question 'What would cause users to spend more time in the app?' shows goal-directed reasoning by working backward from the desired outcome. The question 'What creates those conditions?' demonstrates prerequisite identification by recursively identifying what's needed. The text 'For better recommendations, we would need to improve our machine learning models, which requires better user data collection and analysis' shows recursive backtracking by identifying prerequisites of prerequisites. These demonstrate basic backward chaining but with limited depth.",
  "spans": [[310, 363], [447, 477], [600, 738]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To prove that the triangle ABC is congruent to triangle DEF, I'll work backward from this goal.

For triangles to be congruent, I need to establish one of the congruence criteria: SSS, SAS, ASA, or AAS. The most direct approach would be to use the SSS (Side-Side-Side) criterion, which requires showing that all corresponding sides are equal.

So to achieve the goal of proving congruence, I need to establish:
1. AB = DE
2. BC = EF
3. AC = DF

Starting with the first requirement (AB = DE), what would allow me to conclude this? According to the given information, AB and DE are radii of circles with equal radii, so AB = DE is established.

Moving to the second requirement (BC = EF), what conditions would make this true? The problem states that BC and EF are both tangents to their respective circles from points B and E. For a tangent from an external point to a circle, its length depends on the distance from the point to the center and the circle's radius. Since we know that the distances from B and E to their respective circle centers are equal (given), and the circles have equal radii (given), then BC = EF by the tangent-secant theorem.

For the third requirement (AC = DF), what must be true? A and D are the centers of their respective circles, while C and F are the points of tangency. Since the tangent to a circle is perpendicular to the radius at the point of tangency, AC and DF are both radii of their respective circles. Given that the circles have equal radii, AC = DF is established.

Since all three requirements for SSS congruence have been established by working backward from needed conditions to given facts, triangles ABC and DEF are congruent."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated backward chaining by explicitly starting with the goal (proving triangle congruence), identifying the criteria needed to establish that goal (SSS), working backward to identify the specific requirements for each criterion, and then systematically establishing the prerequisites for each requirement until reaching the given facts.",
  "span_analysis": "Analyzing this text for backward chaining: The phrase 'I'll work backward from this goal' explicitly establishes the backward chaining approach. The text 'So to achieve the goal of proving congruence, I need to establish:' shows goal-directed reasoning by identifying prerequisites. The questions 'what would allow me to conclude this?', 'what conditions would make this true?', and 'what must be true?' demonstrate systematic prerequisite identification by working backward from each requirement to its necessary conditions. These show sophisticated, systematic backward chaining from goal to given facts.",
  "spans": [[61, 95], [344, 410], [492, 529], [687, 724], [1189, 1207]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of backward chaining is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/causal-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
CAUSAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Causal Organization in the Reasoning Process

## Definition
**Causal organization** is the ability to arrange elements through cause-effect relationships. In reasoning traces, causal organization refers to when the participant demonstrates the ability to identify causal connections between events or states, reason about chains of causation, understand causal mechanisms, and evaluate how interventions would propagate through causal systems.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates causal organization:

1. **Causal identification**: Does the participant explicitly identify cause-effect relationships?
   - Look for statements about what causes what, or why something happens
   - Check if the participant distinguishes causes from correlations or coincidences

2. **Causal chains**: Does the participant trace extended chains of causation?
   - Look for multi-step causal reasoning (A causes B, which causes C...)
   - Check if the participant connects distant events through causal intermediaries

3. **Causal mechanisms**: Does the participant explain how causes produce their effects?
   - Look for explanations of the processes linking causes to effects
   - Check if the participant identifies the underlying mechanisms of causation

4. **Counterfactual reasoning**: Does the participant reason about what would happen under different causal conditions?
   - Look for consideration of hypothetical interventions
   - Check if the participant evaluates how changes would propagate through causal systems

## Label Levels

**0 - Absent**: The reasoning trace shows little to no causal organization. The participant describes events or states without establishing cause-effect relationships or shows only superficial attention to causation.

**1 - Partially Present**: The reasoning trace shows some causal organization, but with limited depth or inconsistent application. The participant identifies some cause-effect relationships but may not develop extended causal chains, explain mechanisms in detail, or engage in sophisticated counterfactual reasoning.

**2 - Present**: The reasoning trace shows clear causal organization throughout. The participant consistently identifies cause-effect relationships, traces extended causal chains, explains causal mechanisms, and engages in counterfactual reasoning about interventions.

## Span Identification Instructions

In addition to scoring the overall presence of causal organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate causal organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of causal organization (causal identification, causal chains, causal mechanisms, or counterfactual reasoning)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to causal organization
- **Multiple spans**: Identify all significant instances of causal organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of causal organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates causal organization

### How to Identify Valid Spans
1. **Look for causal language**: Words like "causes," "because," "leads to," "results in," "due to," "triggers," "produces," "brings about," "if...then," "would result in"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant identifying cause-effect relationships or reasoning about causal mechanisms?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether causal organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate causal organization. Describe what specific phrases or sentences show causal identification, causal chains, causal mechanisms, or counterfactual reasoning, and explain why they demonstrate causal organization. This guides you to identify the character indices. Use an empty string "" if no causal organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate causal organization. Use an empty list [] if no spans demonstrate causal organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The water cycle consists of several phases: evaporation, condensation, precipitation, and collection. During evaporation, the sun heats water on Earth's surface, turning it into water vapor. In condensation, water vapor cools and forms clouds. Precipitation occurs when water falls from clouds as rain, snow, sleet, or hail. Collection happens when water returns to bodies of water like oceans, lakes, and rivers, or soaks into the ground to become groundwater."

Annotation:
```json
{{
  "explanation": "The participant describes the phases of the water cycle sequentially but doesn't establish clear causal relationships between phases, explain the mechanisms that drive the transitions, or engage in any causal analysis of the process.",
  "span_analysis": "This text shows no causal organization - it describes sequential phases without establishing cause-effect relationships, explaining causal mechanisms, or reasoning about causal connections between the phases.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "The Great Depression was caused primarily by the stock market crash of 1929, which led to a banking crisis as many banks had invested heavily in stocks. When banks failed, the money supply contracted, causing deflation. This deflation made debts more expensive in real terms, leading businesses to cut costs by laying off workers. Rising unemployment then reduced consumer spending, which further hurt businesses and led to more layoffs in a vicious cycle. Government policies at the time, like raising interest rates and tariffs, made the situation worse instead of better."

Annotation:
```json
{{
  "explanation": "The participant identifies several cause-effect relationships and traces a basic causal chain in the development of the Great Depression, but provides limited explanation of the underlying mechanisms and doesn't engage in counterfactual reasoning about potential interventions.",
  "span_analysis": "Looking for causal organization in this text: The phrase 'The Great Depression was caused primarily by the stock market crash of 1929, which led to a banking crisis' shows causal identification and causal chains. The text 'When banks failed, the money supply contracted, causing deflation' demonstrates causal chains by connecting events. The phrase 'This deflation made debts more expensive in real terms, leading businesses to cut costs' shows continued causal reasoning. These demonstrate basic causal organization through cause-effect identification and simple causal chains.",
  "spans": [[0, 137], [138, 209], [210, 310]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "Let's analyze the causal pathways that led to the Arab Spring uprisings, focusing particularly on Tunisia as the initial case.

The primary causal chain began with long-term structural factors: authoritarian governance created systems with limited political freedoms, which caused public frustration to build over decades. This authoritarian system directly caused economic conditions characterized by corruption and inequality, as political elites captured economic resources without accountability mechanisms.

These economic conditions produced widespread unemployment, particularly among educated youth, through several mechanisms: corruption reduced meritocracy in hiring, patronage networks controlled job access, and state-dominated economies failed to create sufficient private sector opportunities. High youth unemployment specifically caused heightened grievances because it violated the implicit social contract where educational achievement should lead to employment opportunities.

When economic crisis hit globally in 2008, it triggered an intensification of these existing problems through multiple causal pathways: food prices rose, remittances from abroad fell, tourism decreased, and export industries suffered. These factors caused immediate material hardship for populations already under economic stress.

The self-immolation of Mohamed Bouazizi served as the proximate trigger that transformed these conditions into mass mobilization through several causal mechanisms: it crystallized existing grievances into a symbolic act, generated emotional solidarity, and provided a focal point for coordination.

Social media played a crucial causal role by altering the information environment in three ways: it reduced communication costs for organization, circumvented state media controls, and created visibility of protests that caused preference falsification to collapse (people who privately opposed the regime could now see others felt similarly, causing a cascade effect).

The causal relationship between Tunisia's uprising and subsequent movements in other countries operated through demonstration effects - the success in Tunisia showed similar actions were possible elsewhere - and through transnational information flows that transferred protest tactics and frames.

If we consider counterfactual scenarios, several interventions might have altered this causal chain: had governments implemented genuine economic reforms that addressed youth unemployment earlier, the material grievances would have been reduced. If social media had been more effectively controlled by states, the coordination mechanisms would have been weakened. Had early protests been met with concessions rather than repression, escalation might have been prevented through a causal path of grievance reduction."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated causal organization by methodically identifying multiple cause-effect relationships at different levels, tracing complex causal chains with interconnected factors, explicitly explaining the mechanisms linking causes to effects, and engaging in detailed counterfactual reasoning about how different interventions would have altered causal pathways.",
  "span_analysis": "Analyzing this text for causal organization: The phrase 'Let's analyze the causal pathways that led to the Arab Spring uprisings' explicitly establishes causal analysis as the approach. The text 'The primary causal chain began with long-term structural factors:' demonstrates causal chains by identifying the starting point of causation. The phrase 'authoritarian governance created systems with limited political freedoms, which caused public frustration to build over decades' shows causal identification and causal mechanisms by explaining how causes produce effects. These demonstrate sophisticated causal organization through explicit causal analysis and mechanism explanation.",
  "spans": [[0, 71], [128, 193], [194, 321]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of causal organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/compositionality-trace-prompt.txt
# -----------------------------------------------------------------------------
COMPOSITIONALITY_PROMPT = r"""# Annotation Guidelines: Compositionality in the Reasoning Process

## Definition
**Compositionality** is the ability to build complex ideas from simpler components. In reasoning traces, compositionality refers to when the participant demonstrates the ability to decompose complex problems into simpler parts, reason about these parts individually, and then recombine them to address the overall problem.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates compositionality:

1. **Conceptual decomposition**: Does the participant break down complex concepts into simpler components?
   - Look for the identification of constituent elements of a complex idea
   - Check if the participant explains complex notions in terms of more basic concepts

2. **Modular reasoning**: Does the participant reason about components individually before combining them?
   - Look for separate treatment of distinct problem elements
   - Check if the participant addresses subproblems before tackling the whole

3. **Hierarchical construction**: Does the participant build up complex structures from simpler ones?
   - Look for progressive construction of ideas, starting from basic elements
   - Check if the participant creates intermediate concepts that build toward the final solution

4. **Recombination**: Does the participant effectively integrate component solutions?
   - Look for synthesis of individual analyses into a coherent whole
   - Check if the participant ensures consistency when combining partial solutions

## Label Levels

**0 - Absent**: The reasoning trace shows little to no compositionality. The participant approaches problems holistically without breaking them down into component parts or building up complex ideas from simpler ones.

**1 - Partially Present**: The reasoning trace shows some compositionality, but with limited depth or inconsistent application. The participant may partially decompose problems or combine ideas, but doesn't consistently apply compositional principles throughout its reasoning.

**2 - Present**: The reasoning trace shows clear compositionality throughout. The participant systematically breaks down complex concepts, reasons about components separately, builds hierarchical structures, and recombines elements to address the overall problem.

## Span Identification Instructions

In addition to scoring the overall presence of compositionality, you must identify specific spans (text segments) in the reasoning trace that demonstrate compositionality. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of compositionality (decomposition, component reasoning, hierarchical building, or recombination)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to compositionality
- **Multiple spans**: Identify all significant instances of compositionality, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of compositionality are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates compositionality

### How to Identify Valid Spans
1. **Look for compositionality language**: Words like "break down," "decompose," "component," "part," "element," "combine," "integrate," "build up," "construct," "assemble"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant breaking down complex ideas into parts or building up from simpler components?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether compositionality is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate compositionality. Describe what specific phrases or sentences show decomposition, component reasoning, hierarchical building, or recombination, and explain why they demonstrate compositionality. This guides you to identify the character indices. Use an empty string "" if no compositionality is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate compositionality. Use an empty list [] if no spans demonstrate compositionality.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The question asks for the meaning of the poem 'The Road Not Taken' by Robert Frost. This poem is about making choices in life. The narrator comes to a fork in the road and has to decide which path to take. He chooses the one less traveled, and that has made all the difference in his life. The poem is about individuality and how our choices define our lives. So the meaning of the poem is about the importance of making our own choices rather than following the crowd."

Annotation:
```json
{{
  "explanation": "The participant provides a holistic interpretation without breaking down the poem into constituent elements, analyzing specific components, or building up its interpretation from simpler components.",
  "span_analysis": "This text shows no compositionality - it provides a general interpretation without decomposing the poem into parts, analyzing components separately, or building up understanding from simpler elements.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To calculate the total cost of the shopping trip, I need to find the cost of each item and add them together. The apples cost $1.50 per pound, and I'm buying 3 pounds, so that's $1.50 × 3 = $4.50. The bread costs $2.75 per loaf, and I'm buying 2 loaves, so that's $2.75 × 2 = $5.50. The milk costs $3.25 per gallon, and I'm buying 1 gallon, so that's $3.25. Now I'll add these up: $4.50 + $5.50 + $3.25 = $13.25. Therefore, the total cost is $13.25."

Annotation:
```json
{{
  "explanation": "The participant shows basic compositionality by breaking down the total cost calculation into individual item costs and then recombining them, but doesn't demonstrate more complex compositional reasoning beyond this straightforward decomposition and addition.",
  "span_analysis": "Looking for compositionality in this text: The phrase 'I need to find the cost of each item and add them together' shows decomposition by breaking down the total into components. The individual calculations for apples, bread, and milk demonstrate component reasoning by addressing each part separately. The text 'Now I'll add these up: $4.50 + $5.50 + $3.25 = $13.25' shows recombination by integrating the components. These demonstrate basic compositionality through decomposition and recombination.",
  "spans": [[59, 118], [119, 194], [390, 459]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To analyze the economic impact of the proposed policy, I'll break this down into components and then integrate them for an overall assessment.

First, let's examine the direct fiscal effects:
1. Revenue generation: The new tax is estimated to generate $50 million annually
2. Implementation costs: Administrative overhead is projected at $5 million initially, then $2 million annually
Direct fiscal impact: +$48 million in year one, +$48 million in subsequent years

Second, let's consider market effects:
1. Producer response: Likely price increases of 2-3% based on elasticity data
2. Consumer response: Estimated 1.5% reduction in consumption
3. Labor market: Minimal effect (less than 0.1% employment change)
Net market effect: Slight contraction in market size with minimal employment impact

Third, let's analyze distributional consequences:
1. Income groups: Regressive impact with lower-income households spending 0.5% more of disposable income
2. Geographic distribution: Rural communities face 1.2% higher effective burden due to fewer alternatives
3. Sectoral impact: Related industries may see 0.8% reduction in revenue

Now, integrating these components into an overall assessment:
The policy generates substantial revenue ($48M annually) with modest market distortion. However, the regressive nature and uneven geographic impact suggest targeted mitigation measures would be valuable. Given the revenue-to-distortion ratio compared to alternatives, this policy shows moderate economic efficiency but raises equity concerns. The integrated analysis suggests the policy would be most effective if combined with compensatory measures for the most affected groups."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated compositionality by systematically breaking down the analysis into distinct components (fiscal, market, and distributional effects), addressing each component with appropriate subcomponents, and then integrating these separate analyses into a coherent overall assessment with new insights emerging from the combination.",
  "span_analysis": "Analyzing this text for compositionality: The phrase 'I'll break this down into components and then integrate them for an overall assessment' explicitly establishes the compositional approach through decomposition and recombination. The text 'First, let's examine the direct fiscal effects:' shows systematic component reasoning by addressing parts separately. The phrase 'Direct fiscal impact: +$48 million in year one, +$48 million in subsequent years' demonstrates hierarchical building by synthesizing subcomponents. The text 'Second, let's consider market effects:' shows continued decomposition into distinct components. These demonstrate sophisticated compositionality through systematic decomposition, component analysis, and integration.",
  "spans": [[55, 142], [144, 191], [378, 452], [454, 492]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of compositionality is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/conceptual-level-processing-trace-prompt.txt
# -----------------------------------------------------------------------------
CONCEPTUAL_LEVEL_PROCESSING_PROMPT = r"""# Annotation Guidelines: Conceptual-level Processing in the Reasoning Process

## Definition
**Conceptual-level processing** is the ability to reason with abstract concepts before translating to linguistic forms. In reasoning traces, conceptual-level processing refers to when the participant demonstrates an understanding of the deep meaning or abstract structure of a problem beyond surface features, works with high-level concepts rather than just procedural steps, and shows evidence of thinking at a more abstract level than mere linguistic manipulation.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates conceptual-level processing:

1. **Abstract representation**: Does the participant work with abstract concepts rather than just concrete details?
   - Look for identification of underlying principles or patterns
   - Check if the participant represents problems in terms of abstract structures

2. **Meaning-focused reasoning**: Does the participant focus on meaning rather than just linguistic form?
   - Look for reasoning that captures the essence of concepts beyond their verbal expression
   - Check if the participant distinguishes between surface expressions and deeper meaning

3. **Structural understanding**: Does the participant grasp the conceptual structure of the problem?
   - Look for recognition of deep similarities between superficially different problems
   - Check if the participant can identify abstract patterns or schemas

4. **Conceptual transformation**: Does the participant transform problems at a conceptual level?
   - Look for reformulation of problems in more useful conceptual terms
   - Check if the participant can shift between different conceptual frameworks

## Label Levels

**0 - Absent**: The reasoning trace shows little to no conceptual-level processing. The participant works primarily with surface features, focuses on procedural steps without higher-level understanding, or manipulates language without apparent grasp of deeper meaning.

**1 - Partially Present**: The reasoning trace shows some conceptual-level processing, but with limited depth or inconsistent application. The participant occasionally works with abstract concepts or demonstrates some understanding of deeper structures, but doesn't consistently reason at a conceptual level.

**2 - Present**: The reasoning trace shows clear conceptual-level processing throughout. The participant consistently works with abstract concepts, focuses on meaning rather than just form, demonstrates understanding of deep structure, and transforms problems at a conceptual level.

## Span Identification Instructions

In addition to scoring the overall presence of conceptual-level processing, you must identify specific spans (text segments) in the reasoning trace that demonstrate conceptual-level processing. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of conceptual-level processing (abstract representation, meaning-focused reasoning, structural understanding, or conceptual transformation)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to conceptual-level processing
- **Multiple spans**: Identify all significant instances of conceptual-level processing, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of conceptual-level processing are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates conceptual-level processing

### How to Identify Valid Spans
1. **Look for conceptual language**: Words like "abstract," "concept," "principle," "essence," "underlying," "fundamental," "structure," "pattern," "framework," "meaning," "represents"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant working with abstract concepts or deep meaning rather than just surface features?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether conceptual-level processing is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate conceptual-level processing. Describe what specific phrases or sentences show abstract representation, meaning-focused reasoning, structural understanding, or conceptual transformation, and explain why they demonstrate conceptual-level processing. This guides you to identify the character indices. Use an empty string "" if no conceptual-level processing is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate conceptual-level processing. Use an empty list [] if no spans demonstrate conceptual-level processing.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve 15 × 27, I'll use the standard multiplication algorithm. First, I multiply 7 × 5 = 35, so I write down 5 and carry the 3. Then I multiply 7 × 1 = 7, and add the carried 3 to get 10, so I write down 0 and carry the 1. Then I multiply 2 × 5 = 10, and write down 0 and carry the 1. Finally, I multiply 2 × 1 = 2, add the carried 1 to get 3, and write that down. So the answer is 405."

Annotation:
```json
{{
  "explanation": "The participant follows a procedural algorithm without demonstrating any understanding of the conceptual meaning of multiplication, number properties, or alternative representations, focusing entirely on mechanical steps rather than abstract concepts.",
  "span_analysis": "This text shows no conceptual-level processing - it follows mechanical procedural steps without working with abstract concepts, understanding deeper meaning, or demonstrating structural understanding beyond surface operations.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "The problem asks whether a monopolist would increase or decrease production if the marginal cost increases. I'll approach this using the concept of profit maximization, where a rational monopolist produces at the quantity where marginal revenue equals marginal cost. If marginal cost increases, the intersection point of marginal revenue and marginal cost will shift to the left, meaning a lower quantity. This makes intuitive sense because higher costs make producing additional units less profitable. Therefore, the monopolist would decrease production."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some conceptual-level processing by working with economic concepts like profit maximization and the relationship between marginal revenue and cost, but provides a relatively straightforward application without exploring deeper conceptual implications or alternative frameworks.",
  "span_analysis": "Looking for conceptual-level processing in this text: The phrase 'I'll approach this using the concept of profit maximization' shows abstract representation by working with economic concepts. The text 'where a rational monopolist produces at the quantity where marginal revenue equals marginal cost' demonstrates structural understanding of economic principles. These show basic conceptual-level processing through abstract economic concepts.",
  "spans": [[102, 158], [160, 259]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "Let's consider what freedom of speech truly means conceptually, beyond just its legal definition. At its core, this concept involves a tension between individual expression and collective harm. I need to think about this in terms of several abstract frameworks:

First, a rights-based framework: Here, freedom of speech represents an inherent liberty of individuals to express their thoughts without prior restraint. This conceptualizes speech as an extension of autonomy and thought itself.

Second, a consequentialist framework: Here, we might view speech in terms of its outcomes—what harms or benefits does it create? This transforms the concept from an absolute right to a social activity with measurable impacts.

Third, a power dynamics framework: This reconceptualizes speech as a mechanism that can either reinforce or challenge existing power structures. The same formal right can have vastly different practical meanings depending on one's social position.

These frameworks reveal that the superficial question of whether someone should be "allowed" to say something masks deeper conceptual questions about what we mean by freedom, harm, autonomy, and power. The legal doctrine of freedom of speech is merely one manifestation of these underlying conceptual tensions.

From a structural perspective, we can see that freedom of speech shares conceptual features with other liberties that involve balancing individual rights against collective interests—similar to property rights or freedom of movement. This allows us to transfer insights from those domains.

If we reconceptualize speech as a form of action rather than mere expression, the question transforms yet again. This suggests that the conceptual boundaries between speech and conduct are themselves social constructions rather than natural categories."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated conceptual-level processing by examining the abstract meaning of freedom of speech through multiple conceptual frameworks, identifying structural similarities with other concepts, transforming the concept in different ways, and revealing deeper patterns beneath surface manifestations.",
  "span_analysis": "Analyzing this text for conceptual-level processing: The phrase 'Let's consider what freedom of speech truly means conceptually, beyond just its legal definition' shows meaning-focused reasoning by going beyond surface features to deeper meaning. The text 'At its core, this concept involves a tension between individual expression and collective harm' demonstrates abstract representation by identifying underlying conceptual structures. The phrase 'I need to think about this in terms of several abstract frameworks:' shows conceptual transformation by explicitly working with abstract frameworks. These demonstrate sophisticated conceptual-level processing through deep meaning analysis and abstract framework application.",
  "spans": [[0, 97], [98, 193], [194, 261]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of conceptual-level processing is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/context-alignment-trace-prompt.txt
# -----------------------------------------------------------------------------
CONTEXT_ALIGNMENT_PROMPT = r"""# Annotation Guidelines: Context Alignment in the Reasoning Process

## Definition
**Context alignment** is the ability to select appropriate organizational patterns based on context. In reasoning traces, context alignment refers to when the participant demonstrates the ability to adapt its reasoning approach based on the specific problem type, select different organizational structures for different contexts, and flexibly switch between structures as needed.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates context alignment:

1. **Contextual assessment**: Does the participant evaluate the context to determine appropriate structures?
   - Look for explicit recognition of problem type or domain
   - Check if the participant considers what approach would be most suitable for the specific context

2. **Structural adaptation**: Does the participant adapt its organizational approach to fit the context?
   - Look for different reasoning structures used for different problems
   - Check if the participant organizes its thinking differently depending on the task

3. **Multiple structure repertoire**: Does the participant show familiarity with various organizational patterns?
   - Look for evidence that the participant can use different structures (hierarchical, sequential, etc.)
   - Check if the participant selects from a repertoire of approaches rather than using one universal method

4. **Flexible switching**: Does the participant change structures if the context shifts?
   - Look for transitions between organizational patterns when appropriate
   - Check if the participant can adapt its approach if initial structures prove ineffective

## Label Levels

**0 - Absent**: The reasoning trace shows little to no context alignment. The participant applies the same organizational pattern regardless of context, without adapting its approach to the specific problem type.

**1 - Partially Present**: The reasoning trace shows some context alignment, but with limited flexibility or inconsistent application. The participant sometimes adapts its organizational approach based on context, but may not select the most appropriate structure or fully adjust to contextual requirements.

**2 - Present**: The reasoning trace shows clear context alignment throughout. The participant consistently evaluates contexts, selects appropriate organizational patterns, draws from a diverse repertoire of structures, and flexibly adapts its approach as needed.

## Span Identification Instructions

In addition to scoring the overall presence of context alignment, you must identify specific spans (text segments) in the reasoning trace that demonstrate context alignment. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of context alignment (contextual assessment, structural adaptation, multiple structure repertoire, or flexible switching)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to context alignment
- **Multiple spans**: Identify all significant instances of context alignment, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of context alignment are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates context alignment

### How to Identify Valid Spans
1. **Look for context alignment language**: Words like "given the context," "for this type of problem," "appropriate approach," "different strategy," "adapt," "switch," "suitable for," "depending on"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant adapting its approach based on context or selecting appropriate structures?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether context alignment is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate context alignment. Describe what specific phrases or sentences show contextual assessment, structural adaptation, multiple structure repertoire, or flexible switching, and explain why they demonstrate context alignment. This guides you to identify the character indices. Use an empty string "" if no context alignment is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate context alignment. Use an empty list [] if no spans demonstrate context alignment.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve this problem, I'll work through it step by step. First, I need to understand what the question is asking. Then, I'll identify the key information provided. Next, I'll apply the relevant formula. After that, I'll calculate the answer. Finally, I'll check my work to make sure the answer makes sense. Following this approach, I see that the question asks for the force needed to accelerate a 50kg object at 10 m/s². Using Newton's Second Law, F = ma, I can substitute the mass and acceleration: F = 50kg × 10m/s² = 500N. Therefore, the force needed is 500 Newtons."

Annotation:
```json
{{
  "explanation": "The participant applies a generic step-by-step approach without any evidence of assessing the specific context to determine an appropriate structure, adapting its organization to the problem type, or considering alternative organizational patterns.",
  "span_analysis": "This text shows no context alignment - it uses a generic step-by-step approach without assessing the context, adapting organizational structures to the problem type, or showing awareness of different structural approaches.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Since this is a legal analysis question, I'll approach it by examining case precedent and statutory interpretation. The question concerns whether a contract was validly formed between parties A and B.

First, I need to identify the elements required for valid contract formation: offer, acceptance, consideration, and intent to create legal relations. For each element, I'll assess the evidence from the scenario.

Regarding the offer, Party A sent an email on March 3rd stating specific terms for the sale of goods. This constitutes a clear offer under contract law principles.

For acceptance, Party B replied on March 5th saying "I agree to your terms." This appears to be unequivocal acceptance.

Consideration is present as Party A was offering goods in exchange for Party B's payment of $5,000.

The intent to create legal relations can be inferred from the business context and formal language used.

Therefore, based on this analysis of the elements, a valid contract was formed between the parties."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some context alignment by recognizing this is a legal question and using an element-by-element analysis appropriate for contract law, but it applies a fairly standard approach without showing a sophisticated repertoire of structures or adapting its organization beyond the basic domain-appropriate format.",
  "span_analysis": "Looking for context alignment in this text: The phrase 'Since this is a legal analysis question, I'll approach it by examining case precedent and statutory interpretation' shows contextual assessment by recognizing the problem type and selecting an appropriate approach. The text 'First, I need to identify the elements required for valid contract formation:' demonstrates structural adaptation by using an element-by-element structure appropriate for legal analysis. These show basic context alignment through domain-appropriate structure selection.",
  "spans": [[0, 120], [204, 287]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "I notice this is a multifaceted problem with both quantitative and ethical dimensions, so I'll need to adapt my approach for each component.

For the quantitative part—calculating the environmental impact of the factory expansion—I'll use a sequential, computational structure. First, I'll calculate the current emissions (50,000 tons CO₂/year) and project the 30% increase (additional 15,000 tons/year). Then I'll analyze water usage similarly (current 2M gallons/day increasing to 2.6M gallons/day) and estimate impact on local water tables using hydrological principles.

Now switching to the economic analysis, which requires a different structure. Here, a comparative framework is more appropriate. I'll create a table comparing current vs. projected states across multiple variables: jobs (250 current, 325 projected), tax revenue ($3M vs. $4.2M), and infrastructure costs ($1M vs. $1.4M).

For the stakeholder analysis portion, I need to adopt yet another structure—a network approach mapping relationships between different parties. The company, local residents, regulatory bodies, and environmental groups form nodes with competing interests. This network reveals potential conflict points and alliance opportunities that wouldn't be visible in a linear analysis.

When addressing the ethical dimension, I'll transition to a principlist framework examining the case through four core principles: autonomy (community's right to self-determination), beneficence (economic benefits), non-maleficence (environmental harm), and justice (distribution of benefits and burdens). This structure is particularly suited for ethical dilemmas where values conflict.

Finally, for the policy recommendation section, I need to integrate these diverse analyses. Here, a hierarchical structure works best: starting with high-level principles, moving to specific criteria, then to concrete recommendations. At the top level, I'll establish the principle of sustainable development, then develop mid-level criteria like "no irreversible environmental damage," and finally specific recommendations like "approve with conditions requiring carbon offsetting."

If this were purely a business case or purely an environmental science problem, I would have structured my entire approach differently. The context-specific requirements of this mixed problem require this adaptive approach."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated context alignment by explicitly recognizing the multifaceted nature of the problem, deliberately selecting different organizational structures for different components (sequential for quantitative analysis, comparative for economic analysis, network for stakeholders, principlist for ethics, and hierarchical for recommendations), and clearly articulating why each structure is appropriate for its specific context.",
  "span_analysis": "Analyzing this text for context alignment: The phrase 'I notice this is a multifaceted problem with both quantitative and ethical dimensions, so I'll need to adapt my approach for each component' shows contextual assessment and structural adaptation by recognizing different problem dimensions and adapting approach accordingly. The text 'For the quantitative part—calculating the environmental impact of the factory expansion—I'll use a sequential, computational structure' demonstrates multiple structure repertoire by selecting a specific organizational structure appropriate for the quantitative context. These demonstrate sophisticated context alignment through explicit contextual assessment and adaptive structure selection.",
  "spans": [[0, 140], [142, 277]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of context alignment is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/context-awareness-trace-prompt.txt
# -----------------------------------------------------------------------------
CONTEXT_AWARENESS_PROMPT = r"""# Annotation Guidelines: Context Awareness in the Reasoning Process

## Definition
**Context awareness** is the meta-cognitive ability to recognize how the situational context shapes which reasoning strategies and goals are appropriate. This includes understanding what kind of task is being performed (e.g., answering a teacher vs. scientific inquiry), what constraints the context imposes (social, institutional, practical), and adapting one's reasoning approach accordingly.

**Key distinction**: Context awareness is about recognizing "this type of situation calls for this type of reasoning"—not merely considering multiple factors. It's meta-cognitive: awareness of how context determines appropriate reasoning approaches.

## What to Look For

Look for evidence the participant recognizes how context shapes appropriate reasoning:

1. **Task context recognition**: Does the participant identify what type of reasoning task this is and what that implies?
   - Examples: "In an educational context...", "For a practical decision...", "If this is testing logic vs. real-world knowledge..."

2. **Constraint identification**: Does the participant identify situational constraints that affect how to reason?
   - Social constraints: audience, expectations, roles (teacher/student, expert/novice)
   - Institutional constraints: procedures, standards, conventions
   - Practical constraints: time, resources, available information

3. **Strategic adaptation**: Does the participant adapt their reasoning strategy based on the context?
   - Must show actual adaptation, not just listing considerations
   - Examples: "In this context, I should prioritize X over Y", "Given this audience, I'll avoid technical terms"

## Label Levels

**0 - Absent**: No recognition of how context shapes appropriate reasoning. The participant treats the task acontextually.

**1 - Partially Present**: Limited context awareness. The participant may acknowledge context but doesn't meaningfully adapt reasoning, or recognizes only superficial aspects.

**2 - Present**: Clear context awareness. The participant recognizes the situational demands, identifies relevant constraints, and adapts their reasoning approach based on what the context requires.

## Span Identification

Identify text segments where the participant demonstrates context awareness.

**Span format**: [start_index, end_index] using 0-based character indices.

**Selection criteria**:
- The span must show the participant recognizing how context shapes appropriate reasoning
- Include complete sentences or meaningful phrases
- Identify all significant instances
- Prefer non-overlapping spans

**How to identify valid spans**:
- Look for language indicating contextual awareness: "in this context", "given the situation", "for this type of task", "depending on the audience", "this calls for"
- Verify: Does this show the participant adjusting their reasoning approach based on situational demands?

## Output Format

Provide strict JSON with exactly four fields:

1. **"explanation"**: One-sentence explanation of whether context awareness is present
2. **"span_analysis"**: Your reasoning for identifying context-aware spans. Describe what phrases show recognition that context shapes appropriate strategies/goals. Use "" if absent.
3. **"spans"**: List of [start_index, end_index] pairs demonstrating context awareness. Use [] if absent.
4. **"score"**: Integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The question asks whether we should recommend Policy A or Policy B. Policy A costs $10 million and affects 1 million people. Policy B costs $5 million and affects 500,000 people. The cost per person for Policy A is $10, while for Policy B it's also $10. Since both have the same cost per person, I'll recommend Policy A because it helps more people. Therefore, Policy A is the better choice."

Annotation:
```json
{{
  "explanation": "The participant applies mechanical cost-benefit calculation without recognizing how the policy recommendation context should shape the reasoning approach (e.g., considering stakeholders, feasibility, political constraints).",
  "span_analysis": "No context awareness is demonstrated. The participant treats this as acontextual arithmetic without acknowledging this is a policy recommendation requiring consideration of implementation constraints, stakeholder needs, or institutional factors.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To determine if this argument is valid, I need to check the logical structure. This seems like a classroom exercise, so I should show my work step by step. The argument has the form: If P then Q, P, therefore Q. This is modus ponens. Let me verify: Premise 1 states the conditional relationship, Premise 2 affirms the antecedent, and the conclusion follows necessarily. The inference rule of modus ponens is: ((P → Q) ∧ P) → Q. Given the truth table for material implication, when both premises are true, the conclusion must be true. Therefore, this argument is deductively valid by modus ponens."

Annotation:
```json
{{
  "explanation": "The participant shows limited context awareness by recognizing this is a classroom exercise requiring step-by-step work, but treats it as purely formal logic without considering whether a classroom context might value pedagogical explanation over technical formalism.",
  "span_analysis": "The phrase 'This seems like a classroom exercise, so I should show my work step by step' demonstrates minimal context awareness—identifying the task type (classroom) and one constraint (show work). However, the response immediately shifts to formal logical notation and truth tables without considering whether a classroom context requires more accessible explanation. The participant recognizes context superficially but doesn't meaningfully adapt the reasoning approach to be pedagogical rather than technical.",
  "spans": [[79, 155]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "This problem asks whether penguins fly, given 'all birds fly' and 'penguins are birds'. In a formal logic exam context, I would apply the syllogism mechanically: the conclusion 'penguins fly' follows from the premises. However, this appears to be testing critical thinking about when to apply formal rules versus real-world knowledge. The appropriate strategy depends on context. For a logic exam: accept the conclusion follows deductively while noting the premise is empirically false. For assessing real-world reasoning: reject the faulty premise. Since this seems designed to test critical thinking rather than mechanical deduction, I'll prioritize real-world knowledge: the premise 'all birds fly' is false because penguins, ostriches, and emus are flightless."

Annotation:
```json
{{
  "explanation": "The participant demonstrates strong context awareness by explicitly recognizing how different contexts (logic exam vs. critical thinking) demand different reasoning strategies, and adapting accordingly.",
  "span_analysis": "Multiple spans show context awareness. 'In a formal logic exam context, I would apply the syllogism mechanically: the conclusion 'penguins fly' follows from the premises' recognizes task type shapes strategy. 'However, this appears to be testing critical thinking about when to apply formal rules versus real-world knowledge' identifies the actual contextual demand. 'The appropriate strategy depends on context. For a logic exam: accept the conclusion follows deductively while noting the premise is empirically false' explicitly shows how different contexts require different approaches. 'For assessing real-world reasoning: reject the faulty premise. Since this seems designed to test critical thinking rather than mechanical deduction, I'll prioritize real-world knowledge' demonstrates strategic adaptation based on contextual assessment.",
  "spans": [[88, 218], [219, 334], [335, 486], [487, 672]],
  "score": 2
}}
```

Now analyze the reasoning trace below. Provide your professional judgment of whether context awareness is present, regardless of correctness.

Question: {question}

Reasoning trace: {response}

Annotation:
"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/decomposition-and-integration-trace-prompt.txt
# -----------------------------------------------------------------------------
DECOMPOSITION_AND_INTEGRATION_PROMPT = r"""# Annotation Guidelines: Decomposition and Integration in the Reasoning Process

## Definition
**Decomposition and integration** is the ability to break problems into subparts and synthesize solutions. In reasoning traces, decomposition and integration refers to when the participant demonstrates the ability to divide complex problems into simpler components, address each component separately, and then recombine the partial solutions into a coherent overall solution.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates decomposition and integration:

1. **Problem decomposition**: Does the participant break down complex problems into simpler parts?
   - Look for explicit division of problems into subproblems
   - Check if the participant identifies component issues that can be addressed separately

2. **Systematic subproblem handling**: Does the participant address subproblems in an organized manner?
   - Look for methodical treatment of each component
   - Check if the participant solves individual parts using appropriate approaches

3. **Part-whole relationships**: Does the participant maintain awareness of how parts relate to the whole?
   - Look for connections between subproblems and the overall problem
   - Check if the participant keeps track of how components fit together

4. **Solution integration**: Does the participant combine partial solutions into a coherent whole?
   - Look for synthesis that brings component solutions together
   - Check if the participant ensures consistency and compatibility when integrating parts

## Label Levels

**0 - Absent**: The reasoning trace shows little to no decomposition and integration. The participant approaches problems holistically without breaking them down into components or addresses parts without effectively integrating them.

**1 - Partially Present**: The reasoning trace shows some decomposition and integration, but with limited systematicity or incomplete application. The participant may break problems into parts but not address them systematically, or may solve components without effectively integrating them.

**2 - Present**: The reasoning trace shows clear decomposition and integration throughout. The participant consistently breaks complex problems into appropriate components, addresses each component systematically, maintains awareness of part-whole relationships, and effectively integrates partial solutions.

## Span Identification Instructions

In addition to scoring the overall presence of decomposition and integration, you must identify specific spans (text segments) in the reasoning trace that demonstrate decomposition and integration. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of decomposition and integration (problem decomposition, systematic subproblem handling, part-whole relationships, or solution integration)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to decomposition and integration
- **Multiple spans**: Identify all significant instances of decomposition and integration, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of decomposition and integration are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates decomposition and integration

### How to Identify Valid Spans
1. **Look for decomposition and integration language**: Words like "break down," "divide," "decompose," "components," "parts," "subproblems," "integrate," "combine," "synthesize," "bring together," "overall solution"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant breaking problems into parts or combining partial solutions?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether decomposition and integration is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate decomposition and integration. Describe what specific phrases or sentences show problem decomposition, systematic subproblem handling, part-whole relationships, or solution integration, and explain why they demonstrate decomposition and integration. This guides you to identify the character indices. Use an empty string "" if no decomposition and integration is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate decomposition and integration. Use an empty list [] if no spans demonstrate decomposition and integration.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve this quadratic equation, x² - 5x + 6 = 0, I'll use the quadratic formula. The formula states that for an equation in the form ax² + bx + c = 0, the solutions are x = (-b ± √(b² - 4ac))/2a. In this case, a = 1, b = -5, and c = 6. Plugging these values into the formula: x = (5 ± √(25 - 24))/2 = (5 ± √1)/2 = (5 ± 1)/2. So x = 6/2 = 3 or x = 4/2 = 2. Therefore, the solutions to the equation are x = 2 and x = 3."

Annotation:
```json
{{
  "explanation": "The participant applies a standard formula to solve the equation without decomposing the problem into subparts or demonstrating any integration of components, instead using a direct algorithmic approach.",
  "span_analysis": "This text shows no decomposition and integration - it follows a direct algorithmic approach without breaking the problem into parts or integrating components.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To evaluate whether the company should launch the new product line, I'll examine several key factors. First, let's look at market demand. Based on the survey data, there appears to be moderate interest among current customers, with 45% indicating they would "likely" or "very likely" purchase the product.

Next, considering production costs, the initial investment would be $500,000 with ongoing production costs of approximately $15 per unit. Given the proposed selling price of $45, the gross margin would be 67%, which is strong.

For competitive analysis, there are three main competitors in this space, with market shares of 35%, 28%, and 15% respectively. Our product has two unique features that differentiate it from competitors.

Looking at potential risks, there are regulatory changes expected next year that could impact production costs by 10-15%.

Taking all these factors together, the product shows promise but has significant risks. The market demand seems sufficient, the financials look promising, we have competitive differentiation, but regulatory changes create uncertainty. Overall, I would recommend proceeding with the launch but developing contingency plans for the regulatory changes."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some decomposition by breaking the product launch decision into separate components (market demand, costs, competition, risks), but provides a relatively superficial analysis of each component and offers limited integration of these factors into a cohesive evaluation beyond a basic summary judgment.",
  "span_analysis": "Looking for decomposition and integration in this text: The phrase 'I'll examine several key factors' shows problem decomposition by breaking down the decision. The text 'First, let's look at market demand' demonstrates systematic subproblem handling. The phrase 'Next, considering production costs' shows continued decomposition. The text 'Taking all these factors together' demonstrates solution integration by combining the components. These show basic decomposition and integration through systematic component analysis.",
  "spans": [[68, 101], [102, 137], [307, 341], [863, 896]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To design an effective urban transportation plan for this growing city, I'll decompose this complex problem into interconnected components, analyze each separately, and then integrate them into a comprehensive solution.

First, let's break this into major subsystems:
1. Public transit infrastructure
2. Road network optimization
3. Non-motorized transportation options
4. Policy and incentive structures
5. Future growth accommodation
6. Environmental impact management

For public transit infrastructure, I need to further decompose into:
- Rail system (capacity, routing, station placement)
- Bus network (frequency, coverage, dedicated lanes)
- Intermodal connections (transfer points, scheduling coordination)

Analyzing the rail component first: The current single-line system is inadequate for projected growth. Based on population density maps and commuter data, I recommend a three-line system with a central hub. Each line requires specific analysis:
- North-south line: 18 stations, 24km length, servicing the high-density corridor and university
- East-west line: 12 stations, 16km length, connecting residential areas with the business district
- Circular line: 15 stations, 20km length, connecting secondary nodes and relieving pressure on central transfers

For the bus network component: Rather than treating buses as secondary, they should form a complementary grid system. This requires:
- Primary routes: 8 high-frequency corridors with dedicated lanes
- Secondary routes: 24 connector routes with standard frequency
- Neighborhood circulators: 36 routes serving last-mile connections

Moving to the road network optimization component:
- Congestion management: Implement dynamic traffic signaling at 47 key intersections
- Bottleneck elimination: Redesign the 5 most congested interchanges
- Parking strategy: Reduce downtown parking by 20%, increase peripheral parking by 35%

For non-motorized transportation:
- Bicycle infrastructure: 85km of protected lanes forming a complete network
- Pedestrian improvements: Walkability enhancements in 12 identified districts
- Micromobility: Designated zones for scooters and bikeshare stations

Now, integrating these components requires addressing cross-system dependencies:
1. Temporal coordination: Phasing construction to minimize disruption (detailed 5-year timeline attached)
2. Spatial integration: Ensuring physical connections between modes (28 key transfer points identified)
3. Operational synchronization: Creating unified scheduling and payment systems
4. Policy coherence: Aligning incentives across transportation modes

The integrated plan demonstrates several emergent properties not visible in the component analysis:
- 23% projected reduction in total commute times through system synergies
- 18% increase in public transit modal share due to network effects
- $47M annual operational savings through optimized connections

This decomposition-integration approach reveals that the optimal solution isn't simply maximizing any single component, but rather finding the balance that creates a cohesive, efficient transportation ecosystem."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated decomposition and integration by systematically breaking down the complex transportation problem into hierarchical components and sub-components, methodically analyzing each part with appropriate approaches, maintaining clear awareness of how components interact, and then comprehensively integrating the solutions with explicit attention to cross-system dependencies and emergent properties.",
  "span_analysis": "Analyzing this text for decomposition and integration: The phrase 'I'll decompose this complex problem into interconnected components, analyze each separately, and then integrate them into a comprehensive solution' explicitly establishes the decomposition and integration approach. The text 'First, let's break this into major subsystems:' demonstrates systematic problem decomposition by breaking down the complex problem into manageable parts. These demonstrate sophisticated decomposition and integration through explicit systematic breakdown and integration planning.",
  "spans": [[72, 219], [221, 267]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of decomposition and integration is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/forward-chaining-trace-prompt.txt
# -----------------------------------------------------------------------------
FORWARD_CHAINING_PROMPT = r"""# Annotation Guidelines: Forward Chaining in the Reasoning Process

## Definition
**Forward chaining** is the ability to start with initial conditions and work toward goals. In reasoning traces, forward chaining refers to when the participant demonstrates the ability to begin with given information, apply logical operations to derive new information, and progressively advance toward a solution through a series of forward inference steps.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates forward chaining:

1. **Initial state utilization**: Does the participant begin reasoning with given information?
   - Look for starting points based on provided facts or premises
   - Check if the participant identifies what is known at the outset

2. **Progressive inference**: Does the participant derive new information from established facts?
   - Look for steps that generate additional knowledge from what's already known
   - Check if the participant extends its knowledge state through logical operations

3. **Forward direction**: Does the participant work from causes to effects or premises to conclusions?
   - Look for reasoning that follows a natural progression from given to derived
   - Check if the participant advances from initial conditions toward goals

4. **Incremental advancement**: Does the participant build solutions step by step in a forward direction?
   - Look for gradual progress toward the solution through successive steps
   - Check if each step builds on previous steps to move closer to the goal

## Label Levels

**0 - Absent**: The reasoning trace shows little to no forward chaining. The participant doesn't start with initial conditions and work forward, instead using other approaches like backward reasoning from goals or lateral thinking.

**1 - Partially Present**: The reasoning trace shows some forward chaining, but with limited systematicity or incomplete application. The participant sometimes works from initial conditions toward goals, but may not consistently apply forward inference or may mix forward chaining with other approaches.

**2 - Present**: The reasoning trace shows clear forward chaining throughout. The participant consistently begins with initial conditions, derives new information through forward inference, works in a forward direction from given to unknown, and advances incrementally toward solutions.

## Span Identification Instructions

In addition to scoring the overall presence of forward chaining, you must identify specific spans (text segments) in the reasoning trace that demonstrate forward chaining. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of forward chaining (initial state utilization, progressive inference, forward direction, or incremental advancement)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to forward chaining
- **Multiple spans**: Identify all significant instances of forward chaining, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of forward chaining are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates forward chaining

### How to Identify Valid Spans
1. **Look for forward chaining language**: Words like "given that," "since we know," "from this we can derive," "this leads to," "building on," "therefore," "consequently," "starting with"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant working forward from given information to derive new conclusions?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether forward chaining is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate forward chaining. Describe what specific phrases or sentences show initial state utilization, progressive inference, forward direction, or incremental advancement, and explain why they demonstrate forward chaining. This guides you to identify the character indices. Use an empty string "" if no forward chaining is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate forward chaining. Use an empty list [] if no spans demonstrate forward chaining.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve for x in the equation 2x + 3 = 11, I need to isolate x. I know that x should equal 4, because 2(4) + 3 = 8 + 3 = 11. To verify this, I could subtract 3 from both sides to get 2x = 8, and then divide both sides by 2 to get x = 4. Therefore, x = 4 is the solution."

Annotation:
```json
{{
  "explanation": "The participant guesses the answer first and then works backward to verify it, rather than starting with the initial equation and working forward through a sequence of operations to derive the solution.",
  "span_analysis": "This text shows no forward chaining - the participant jumps to the answer ('I know that x should equal 4') and then works backward to verify, rather than starting with the given equation and working forward through logical steps.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To determine whether the defendant is liable for negligence, I need to analyze the four elements: duty, breach, causation, and damages.

First, did the defendant owe a duty of care? As a business owner opening premises to the public, the defendant had a duty to maintain safe conditions for customers.

Second, was there a breach of that duty? The evidence shows that the defendant failed to fix a leaking refrigerator unit despite being notified about it twice in the week before the incident. This likely constitutes a breach of the duty of care.

Third, was this breach the cause of the injury? The plaintiff slipped on water that had accumulated from the leaking unit, establishing both actual and proximate causation.

Finally, were there damages? The plaintiff suffered a broken wrist requiring surgery and rehabilitation, constituting clear damages.

Since all four elements are present, the defendant is likely liable for negligence."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some forward chaining by working sequentially through the elements of negligence from initial conditions (facts of the case) to a conclusion, but follows a relatively standard legal framework without requiring sophisticated forward inference beyond direct application of elements to facts.",
  "span_analysis": "Looking for forward chaining in this text: The phrase 'First, did the defendant owe a duty of care?' shows initial state utilization by starting with given facts. The text 'Second, was there a breach of that duty?' demonstrates incremental advancement by building on the previous step. The phrase 'Third, was this breach the cause of the injury?' shows progressive inference by deriving new conclusions from established facts. The text 'Since all four elements are present, the defendant is likely liable for negligence' demonstrates forward direction by reaching a conclusion based on the accumulated evidence. These show basic forward chaining through systematic progression.",
  "spans": [[130, 174], [297, 343], [553, 602], [771, 851]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To determine how the ecosystem will respond to the introduced species, I'll start with the initial conditions and work forward through a chain of ecological relationships.

Beginning with what we know: the introduced species is a mid-sized herbivore with high reproductive capacity, no natural predators in the new environment, and a preference for eating native grasses. The current ecosystem has three native herbivores (small rodent, medium ungulate, large grazer), two predators (mid-sized canid, large feline), and four main plant communities (grassland, shrubland, riparian, forest edge).

From these initial conditions, I can infer the first-order effects: The introduced herbivore will consume native grasses, reducing their biomass. Given the lack of predation and high reproductive capacity, the population will grow rapidly in the short term.

This allows me to deduce second-order effects: Decreased grass biomass will reduce food availability for the native large grazer and medium ungulate, which depend heavily on these grasses. Competition will increase, likely leading to population declines in these native herbivores. The small rodent may be less affected as it primarily feeds on seeds and shrub vegetation.

These changes enable prediction of third-order effects: Declining populations of the medium ungulate and large grazer will reduce prey availability for both predators. The mid-sized canid may partially adapt by increasing predation on small rodents, but the large feline, which specializes in the larger prey, will face significant food stress.

Fourth-order effects become apparent: Reduced grazing pressure on shrubs (from declining native herbivores) combined with decreased grass coverage will accelerate conversion of grassland to shrubland. This habitat change will favor the small rodent, potentially leading to its population increase despite greater predation pressure from the canid.

Fifth-order effects emerge: Expanding shrubland will reduce habitat quality for the large grazer, reinforcing its decline. Increasing small rodent populations may eventually provide enough alternative prey to partially sustain the mid-sized canid, but likely not the large feline, which may experience significant population reduction.

By continuing this forward-chaining process through multiple levels of ecological interactions, I can project the long-term outcome: a transformed ecosystem with reduced biodiversity, dominated by shrubland rather than grassland, supporting primarily the introduced herbivore and small rodent, with reduced predator populations and potential local extinction of the large feline predator if it cannot adapt to alternative prey sources."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated forward chaining by explicitly starting with initial ecological conditions, systematically deriving new information through multiple orders of causal inference (first-order through fifth-order effects), consistently working in a forward causal direction, and incrementally building understanding of ecosystem changes through successive logical steps.",
  "span_analysis": "Analyzing this text for forward chaining: The phrase 'I'll start with the initial conditions and work forward through a chain of ecological relationships' explicitly establishes forward chaining by starting with given information and working forward. The text 'Beginning with what we know: the introduced species is a mid-sized herbivore with high reproductive capacity' shows initial state utilization by identifying starting conditions. These demonstrate sophisticated forward chaining through explicit forward progression from initial conditions.",
  "spans": [[71, 171], [173, 281]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of forward chaining is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/goal-management-trace-prompt.txt
# -----------------------------------------------------------------------------
GOAL_MANAGEMENT_PROMPT = r"""# Annotation Guidelines: Goal Management in the Reasoning Process

## Definition
**Goal management** is the ability to establish, maintain, and adjust goals throughout the reasoning process. In reasoning traces, goal management refers to when the participant explicitly formulates clear objectives, tracks progress toward these goals, handles subgoals effectively, and adjusts goals when necessary.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates goal management:

1. **Goal articulation**: Does the participant explicitly formulate and state their goals?
   - Look for statements like "The goal is to determine...", "We need to find...", or "Our objective is to..."
   - Check if the participant clearly defines what they are trying to accomplish

2. **Subgoal creation**: Does the participant break down complex goals into manageable subgoals?
   - Look for systematic decomposition of main objectives into intermediate steps
   - Check if the participant creates a logical hierarchy of goals and subgoals

3. **Goal tracking**: Does the participant track progress toward their goals throughout the reasoning?
   - Look for references back to established goals during the reasoning process
   - Check if the participant acknowledges when subgoals have been achieved

4. **Goal adjustment**: Does the participant revise or refine goals when necessary?
   - Look for recognition when initial goals need modification
   - Check if the participant can pivot to new or adjusted goals based on insights gained during reasoning

## Label Levels

**0 - Absent**: The reasoning trace shows no evidence of goal management. The participant proceeds without clearly articulating goals, breaking down complex objectives, tracking progress, nor adjusting goals when needed.

**1 - Partially Present**: The reasoning trace shows some evidence of goal management, but it is limited or inconsistent. The participant may establish initial goals but fail to track them effectively, create subgoals without a clear hierarchy or purpose, or neglect to adjust goals when appropriate.

**2 - Present**: The reasoning trace shows clear evidence of goal management throughout. The participant explicitly formulates goals, breaks them down into appropriate subgoals, consistently tracks progress, and demonstrates the ability to adjust goals when necessary.

## Span Identification Instructions

In addition to scoring the overall presence of goal management, you must identify specific spans (text segments) in the reasoning trace that demonstrate goal management. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of goal management (goal articulation, subgoal creation, goal tracking, or goal adjustment)
- **Completeness**: Include enough context to understand the goal management being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to goal management
- **Multiple spans**: Identify all significant instances of goal management, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of goal management are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of goal management it demonstrates (goal articulation, subgoal creation, goal tracking, or goal adjustment)
- How it contributes to the overall goal management score

### How to Identify Valid Spans
1. **Look for goal management language**: Words like "goal," "objective," "aim," "target," "plan," "strategy," "first I need to," "my approach is," "step by step"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant setting, tracking, or adjusting goals in their reasoning process?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether goal management is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate goal management. Describe what specific phrases or sentences show goal articulation, subgoal decomposition, progress tracking, or goal adjustment, and explain why they demonstrate goal management. This guides you to identify the character indices. Use an empty string "" if no goal management is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate goal management. Use an empty list [] if no spans demonstrate goal management.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The equation is 3x + 2y = 12 and 2x - y = 5. We can solve for x in the second equation: 2x - y = 5, so 2x = 5 + y, thus x = (5 + y)/2. Now substitute this into the first equation: 3((5 + y)/2) + 2y = 12. Simplifying: 3(5 + y)/2 + 2y = 12. Further simplification gives us 15/2 + 3y/2 + 2y = 12. Combining like terms: 15/2 + 7y/2 = 12. Therefore, 7y/2 = 12 - 15/2 = 24/2 - 15/2 = 9/2. So y = 9/2 ÷ 7/2 = 9/7. Going back to x = (5 + y)/2, we get x = (5 + 9/7)/2 = (35/7 + 9/7)/2 = 44/14 = 22/7."

Annotation:
```json
{{
  "explanation": "The participant simply executes a solution procedure without explicitly articulating goals, creating subgoals, tracking progress toward objectives, or demonstrating any clear goal management.",
  "span_analysis": "This text shows no goal management - it's a straightforward algebraic solution without any explicit goal articulation, subgoal decomposition, progress tracking, or goal adjustment.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To find the area of the irregular polygon, I'll need to divide it into simpler shapes. To calculate the total area, I'll first split the shape into a rectangle and a triangle. For the rectangle, I'll calculate the area using length × width. The rectangle has dimensions 5m × 3m, so its area is 15m^2. For the triangle, I'll use the formula (1/2) × base × height. The triangle has base 3m and height 4m, so its area is 6m^2. Now, adding the two areas: 15m^2 + 6m^2 = 21m^2. The total area of the irregular polygon is 21m^2."

Annotation:
```json
{{
  "explanation": "The participant briefly states an overall goal and creates implicit subgoals by breaking the problem into parts, but doesn't consistently track progress toward these goals or demonstrate sophisticated goal management throughout the reasoning process.",
  "span_analysis": "Looking for goal management in this text: The phrase 'To find the area of the irregular polygon, I'll need to divide it into simpler shapes' shows goal articulation and strategy planning. The text 'To calculate the total area, I'll first split the shape into a rectangle and a triangle' demonstrates subgoal decomposition. However, the execution is straightforward without explicit progress tracking or goal adjustment.",
  "spans": [[0, 88], [89, 178]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "My goal is to determine whether the given chemical reaction will be spontaneous under standard conditions. To achieve this, I need to calculate the Gibbs free energy (ΔG°) using the formula ΔG° = ΔH° - TΔS°. This requires three subgoals: (1) calculate the enthalpy change (ΔH°), (2) calculate the entropy change (ΔS°), and (3) combine these values at the given temperature.

For the first subgoal, I'll calculate ΔH° using standard enthalpies of formation... [calculation follows]
ΔH° = -92.4 kJ/mol. First subgoal completed.

For the second subgoal, I need to determine ΔS°... [calculation follows]
ΔS° = 65.3 J/(mol·K). Second subgoal completed.

For the third subgoal, I'll calculate ΔG° at 298K... [calculation follows]
ΔG° = -92.4 kJ/mol - (298 K)(65.3 J/(mol·K))(1 kJ/1000 J)
ΔG° = -92.4 kJ/mol - 19.5 kJ/mol
ΔG° = -111.9 kJ/mol

Now I've achieved my primary goal: Since ΔG° is negative (-111.9 kJ/mol), the reaction is spontaneous under standard conditions.

Wait, I should verify my work. Looking again at the entropy calculation, I notice I made an error in accounting for the phase change. Let me recalculate the entropy change... [recalculation follows]
Corrected ΔS° = 43.2 J/(mol·K)

I need to adjust my final calculation for the primary goal:
ΔG° = -92.4 kJ/mol - (298 K)(43.2 J/(mol·K))(1 kJ/1000 J)
ΔG° = -92.4 kJ/mol - 12.9 kJ/mol
ΔG° = -105.3 kJ/mol

The reaction remains spontaneous, but with a corrected ΔG° value of -105.3 kJ/mol."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated goal management by explicitly articulating a primary goal, methodically breaking it down into logical subgoals, tracking progress by noting the completion of each subgoal, and showing goal adjustment by identifying an error and revising calculations to better achieve the primary objective.",
  "span_analysis": "Analyzing this text for goal management: The opening 'My goal is to determine whether the given chemical reaction will be spontaneous under standard conditions' shows explicit goal articulation. The text 'This requires three subgoals: (1) calculate the enthalpy change (ΔH°), (2) calculate the entropy change (ΔS°), and (3) combine these values at the given temperature' demonstrates systematic subgoal decomposition. The phrase 'For the first subgoal, I'll calculate ΔH°' shows progress tracking and subgoal execution. The text 'Now I've achieved my primary goal:' demonstrates goal completion tracking. The phrase 'I need to adjust my final calculation for the primary goal:' shows goal adjustment and revision. These demonstrate comprehensive goal management across all dimensions.",
  "spans": [[0, 106], [207, 372], [374, 415], [835, 869], [1196, 1255]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of goal management is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/hierarchical-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
HIERARCHICAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Hierarchical Organization in the Reasoning Process

## Definition
**Hierarchical organization** is the ability to arrange concepts in nested, tree-like structures with parent-child relationships. In reasoning traces, hierarchical organization refers to when the participant demonstrates the ability to organize information in multiple levels of abstraction, establish clear parent-child relationships between concepts, and navigate between different levels of the hierarchy during reasoning.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates hierarchical organization:

1. **Multilevel structure**: Does the participant organize information at different levels of abstraction?
   - Look for clear distinctions between general principles and specific instances
   - Check if the participant structures information with higher and lower levels

2. **Parent-child relationships**: Does the participant establish clear hierarchical relationships between concepts?
   - Look for explicit nesting of concepts within broader categories
   - Check if the participant shows how specific elements relate to more general ones

3. **Top-down reasoning**: Does the participant work from general principles to specific applications?
   - Look for starting with high-level concepts and deriving specific conclusions
   - Check if the participant applies general rules to particular cases

4. **Bottom-up integration**: Does the participant aggregate specific details into higher-level concepts?
   - Look for synthesis of specific observations into general patterns
   - Check if the participant builds up abstract concepts from concrete examples

## Label Levels

**0 - Absent**: The reasoning trace shows little to no hierarchical organization. The participant treats concepts at a single level without establishing clear parent-child relationships or distinguishing between levels of abstraction.

**1 - Partially Present**: The reasoning trace shows some hierarchical organization, but with limited depth or inconsistent application. The participant occasionally distinguishes between levels of abstraction or establishes some parent-child relationships, but doesn't consistently maintain a hierarchical structure.

**2 - Present**: The reasoning trace shows clear hierarchical organization throughout. The participant consistently organizes information at multiple levels of abstraction, establishes clear parent-child relationships, and effectively navigates between different levels of the hierarchy during reasoning.

## Span Identification Instructions

In addition to scoring the overall presence of hierarchical organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate hierarchical organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of hierarchical organization (multilevel abstraction, parent-child relationships, level navigation, or hierarchical categorization)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to hierarchical organization
- **Multiple spans**: Identify all significant instances of hierarchical organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of hierarchical organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates hierarchical organization

### How to Identify Valid Spans
1. **Look for hierarchical organization language**: Words like "at a higher level," "more specifically," "subcategory," "parent," "child," "nested," "broader," "narrower," "general," "specific," "overarching"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant organizing information in hierarchical levels or establishing parent-child relationships?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether hierarchical organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate hierarchical organization. Describe what specific phrases or sentences show multilevel abstraction, parent-child relationships, level navigation, or hierarchical categorization, and explain why they demonstrate hierarchical organization. This guides you to identify the character indices. Use an empty string "" if no hierarchical organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate hierarchical organization. Use an empty list [] if no spans demonstrate hierarchical organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The French Revolution was a period of radical social and political upheaval in France from 1789 to 1799. It was partly caused by financial problems, social inequality, and Enlightenment ideas. During this time, the monarchy was overthrown, and a republic was established. There was also the Reign of Terror, where many people were executed. Napoleon eventually took power at the end of this period. The revolution had lasting effects on French society and influenced other revolutionary movements around the world."

Annotation:
```json
{{
  "explanation": "The participant presents information as a flat sequence of facts without organizing them into any hierarchical structure, establishing parent-child relationships, or distinguishing between different levels of abstraction.",
  "span_analysis": "This text shows no hierarchical organization - it presents historical facts in a linear sequence without establishing multilevel abstraction, parent-child relationships, or hierarchical categorization.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Let's classify this animal based on its characteristics. It has fur, mammary glands, and is warm-blooded, so it belongs to the class Mammalia. Within mammals, it has powerful legs for jumping and a pouch for carrying young, which places it in the order Diprotodontia. More specifically, its large hind legs and hopping locomotion indicate it's part of the family Macropodidae. Given its size and distinctive features, it appears to be a kangaroo, genus Macropus. Based on its large size and coloration patterns, it's likely a red kangaroo, Macropus rufus."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some hierarchical organization by working through taxonomic levels (class, order, family, genus, species), but doesn't clearly establish the relationships between these levels or consistently navigate between general principles and specific attributes.",
  "span_analysis": "Looking for hierarchical organization in this text: The phrase 'so it belongs to the class Mammalia' shows basic hierarchical categorization. The text 'Within mammals, it has powerful legs for jumping and a pouch for carrying young, which places it in the order Diprotodontia' demonstrates level navigation from broader to narrower categories. The phrase 'More specifically, its large hind legs and hopping locomotion indicate it's part of the family Macropodidae' shows multilevel abstraction by moving to a more specific level. These demonstrate basic hierarchical organization through taxonomic classification.",
  "spans": [[109, 149], [151, 273], [275, 395]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To understand the political structure of the United States, I'll analyze it as a hierarchical system with multiple levels.

At the highest level, we have the federal constitutional republic as the overarching political structure. This top level establishes the foundational principles of governance through the Constitution.

Within this top level, we can identify three main branches forming the next level down:
1. Legislative branch (Congress)
2. Executive branch (Presidency)
3. Judicial branch (Supreme Court and federal courts)

Each of these branches can be further decomposed into constituent elements:

The legislative branch divides into:
- Senate (upper chamber)
  - 100 senators (2 per state)
  - Led by the Vice President as President of the Senate
- House of Representatives (lower chamber)
  - 435 voting members (proportional to state population)
  - Led by the Speaker of the House

The executive branch hierarchically organizes as:
- President (head of executive)
  - Vice President
  - Cabinet departments (15 major departments)
    - Each headed by a Secretary (e.g., Secretary of State)
      - Each department containing multiple agencies and bureaus

The judicial branch structures as:
- Supreme Court (highest court)
  - 9 justices
- Circuit Courts of Appeals (intermediate level)
  - 13 appellate circuits
- District Courts (trial level)
  - 94 federal judicial districts

Beneath the federal level, we have the state level of government, each with its own hierarchical structure mirroring aspects of the federal system with:
- State executive (Governor)
- State legislature (typically bicameral)
- State judiciary

States are further divided into counties or county-equivalents, which contain:
- Municipal governments (cities, towns)
  - Some with their own hierarchical structures (mayor, city council)

This multilevel hierarchical organization allows for both top-down implementation of national policies and bottom-up representation of local interests, creating a complex system of nested governance structures with defined relationships between levels."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated hierarchical organization by systematically structuring information across multiple nested levels (federal → branches → specific institutions → components), clearly establishing parent-child relationships throughout, and navigating effectively between general principles and specific instances.",
  "span_analysis": "Analyzing this text for hierarchical organization: The phrase 'I'll analyze it as a hierarchical system with multiple levels' explicitly establishes the hierarchical approach. The text 'At the highest level, we have the federal constitutional republic as the overarching political structure' shows multilevel abstraction by establishing the top level. The phrase 'Within this top level, we can identify three main branches forming the next level down:' demonstrates clear parent-child relationships and level navigation. These demonstrate sophisticated hierarchical organization through explicit multilevel structuring and systematic level navigation.",
  "spans": [[60, 122], [124, 229], [326, 413]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of hierarchical organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/knowledge-structure-alignment-trace-prompt.txt
# -----------------------------------------------------------------------------
KNOWLEDGE_STRUCTURE_ALIGNMENT_PROMPT = r"""# Annotation Guidelines: Knowledge Structure Alignment in the Reasoning Process

## Definition
**Knowledge structure alignment** is the ability to match reasoning organization to domain knowledge structure. In reasoning traces, knowledge structure alignment refers to when the participant demonstrates the ability to organize its reasoning in ways that reflect how knowledge is structured in the relevant domain, leverage domain-specific organizational patterns, and adapt its reasoning approach to align with the natural organization of the relevant knowledge.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates knowledge structure alignment:

1. **Domain-appropriate organization**: Does the participant organize reasoning in ways that match how knowledge is structured in the domain?
   - Look for reasoning structures that reflect domain conventions
   - Check if the participant uses organizational patterns typical of experts in the field

2. **Terminology and conceptual frameworks**: Does the participant use domain-specific frameworks and terminology?
   - Look for specialized vocabulary and concepts from the relevant field
   - Check if the participant organizes ideas according to established conceptual structures in the domain

3. **Field-specific reasoning patterns**: Does the participant employ reasoning approaches typical of the domain?
   - Look for thought processes that reflect how experts in the field think
   - Check if the participant uses inference patterns or heuristics appropriate to the domain

4. **Alignment with knowledge organization**: Does the participant structure its approach to match how the relevant knowledge is organized?
   - Look for reasoning that mirrors the inherent structure of the domain knowledge
   - Check if the participant leverages natural divisions or relationships in the knowledge area

## Label Levels

**0 - Absent**: The reasoning trace shows little to no knowledge structure alignment. The participant applies generic reasoning patterns without adapting to domain-specific knowledge structures.

**1 - Partially Present**: The reasoning trace shows some knowledge structure alignment, but with limited depth or inconsistent application. The participant occasionally organizes reasoning in domain-appropriate ways or uses some field-specific patterns, but doesn't fully align with the knowledge structure of the domain.

**2 - Present**: The reasoning trace shows clear knowledge structure alignment throughout. The participant consistently organizes reasoning to match domain knowledge structures, uses appropriate terminology and conceptual frameworks, employs field-specific reasoning patterns, and aligns its approach with the natural organization of domain knowledge.

## Span Identification Instructions

In addition to scoring the overall presence of knowledge structure alignment, you must identify specific spans (text segments) in the reasoning trace that demonstrate knowledge structure alignment. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of knowledge structure alignment (domain-specific organization, field-appropriate frameworks, expert reasoning patterns, or disciplinary terminology)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to knowledge structure alignment
- **Multiple spans**: Identify all significant instances of knowledge structure alignment, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of knowledge structure alignment are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates knowledge structure alignment

### How to Identify Valid Spans
1. **Look for domain-specific language**: Technical terms, field-specific concepts, disciplinary frameworks, expert terminology
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant using domain-appropriate knowledge structures or expert reasoning patterns?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether knowledge structure alignment is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate knowledge structure alignment. Describe what specific phrases or sentences show domain-specific organization, field-appropriate frameworks, expert reasoning patterns, or disciplinary terminology, and explain why they demonstrate knowledge structure alignment. This guides you to identify the character indices. Use an empty string "" if no knowledge structure alignment is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate knowledge structure alignment. Use an empty list [] if no spans demonstrate knowledge structure alignment.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve for x in the equation 2x² + 3x - 5 = 0, I'll use the quadratic formula. The formula states that for an equation ax² + bx + c = 0, the solutions are x = (-b ± √(b² - 4ac))/2a. In this case, a = 2, b = 3, and c = -5. Substituting these values: x = (-3 ± √(9 - 4(2)(-5)))/2(2) = (-3 ± √(9 + 40))/4 = (-3 ± √49)/4 = (-3 ± 7)/4. So x = (-3 + 7)/4 = 4/4 = 1 or x = (-3 - 7)/4 = -10/4 = -2.5. Therefore, the solutions are x = 1 and x = -2.5."

Annotation:
```json
{{
  "explanation": "The participant applies a standard mathematical procedure without demonstrating any alignment with domain-specific knowledge structures, specialized conceptual frameworks, or reasoning patterns beyond the routine application of a formula.",
  "span_analysis": "This text shows no knowledge structure alignment - it follows a routine formula application without demonstrating domain-specific organization, expert reasoning patterns, or specialized frameworks.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "In analyzing this poem, I'll examine its form and content to understand its meaning. The poem follows a sonnet structure with 14 lines in iambic pentameter, divided into an octave (first 8 lines) presenting a problem and a sestet (last 6 lines) offering a resolution. This form was popularized by Petrarch and later adapted by Shakespeare, though this example more closely follows the Petrarchan model with its clear volta (turn) at line 9.

The rhyme scheme is ABBAABBA CDECDE, confirming it as a Petrarchan sonnet. Thematically, the poem deals with the transience of beauty and time's inevitable progress, common concerns in the sonnet tradition. The imagery shifts from natural elements in the octave to more abstract concepts in the sestet, creating the contrast typical of this form. The final couplet delivers the poem's resolution, suggesting that art preserves what time destroys."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some knowledge structure alignment by organizing its literary analysis according to standard poetic forms (sonnet structure, rhyme scheme), using domain-specific terminology (volta, octave, sestet), and recognizing thematic patterns typical of the sonnet tradition, but doesn't fully develop the interpretive frameworks that would show deeper alignment with literary analysis methods.",
  "span_analysis": "Looking for knowledge structure alignment in this text: The phrase 'The poem follows a sonnet structure with 14 lines in iambic pentameter, divided into an octave (first 8 lines) presenting a problem and a sestet (last 6 lines) offering a resolution' shows domain-specific organization using literary analysis frameworks and expert terminology. The text 'The rhyme scheme is ABBAABBA CDECDE, confirming it as a Petrarchan sonnet' demonstrates field-appropriate frameworks by using technical literary analysis methods. The phrase 'common concerns in the sonnet tradition' shows expert reasoning patterns by recognizing domain-specific patterns. These demonstrate knowledge structure alignment through literary analysis expertise.",
  "spans": [[85, 267], [442, 516], [608, 647]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To diagnose the patient's condition, I'll approach this using the standard differential diagnosis methodology employed in clinical medicine.

First, I'll organize the information according to the classic medical history structure: presenting complaint, history of present illness, past medical history, family history, social history, and review of systems. The patient presents with acute right lower quadrant abdominal pain (presenting complaint) that began periumbilically 24 hours ago before migrating (history of present illness). The associated symptoms include nausea, vomiting, and fever. Past medical history is unremarkable, and family history is negative for inflammatory bowel disease or cancer.

Following the standard clinical reasoning pattern, I'll develop a problem representation: "A 22-year-old previously healthy male with acute-onset periumbilical pain migrating to the right lower quadrant, accompanied by nausea, vomiting, anorexia, and low-grade fever."

Next, I'll employ semantic qualifiers, as is typical in medical reasoning, to characterize the problem: acute (vs. chronic), severe (vs. mild), migratory (vs. fixed), and associated with systemic symptoms.

For the differential diagnosis, I'll use the anatomic framework that organizes abdominal pathology by structures in the right lower quadrant: appendix, terminal ileum, cecum, right ovary/fallopian tube (not applicable in this male patient), ureter, and abdominal wall. This anatomic organization is fundamental to clinical reasoning about abdominal pain.

Given the classic progression from periumbilical to right lower quadrant pain, I'll prioritize appendicitis at the top of my differential, following the "common conditions are common" heuristic used in clinical reasoning. Alternative diagnoses include mesenteric adenitis, Crohn's disease flare, cecal diverticulitis, and kidney stone, organized by probability based on the clinical presentation.

For diagnostic evaluation, I'll follow the evidence-based sequence: laboratory studies first (CBC with differential, looking for leukocytosis with left shift typical of appendicitis), then imaging (ultrasound or CT scan), reflecting the standard clinical pathway for suspected appendicitis.

The treatment recommendation will follow the surgical decision tree for appendicitis: NPO status, IV fluids, antibiotics, and appendectomy, either laparoscopic or open depending on surgical expertise and patient factors."

Annotation:
```json
{{
  "explanation": "The participant demonstrates exceptional knowledge structure alignment by organizing its reasoning according to the precise structures used in clinical medicine (medical history framework, problem representation, anatomic organization of differential diagnosis), employing domain-specific terminology and conceptual frameworks (semantic qualifiers, evidence-based diagnostic sequence), using clinical reasoning patterns (common conditions are common heuristic), and structuring the entire approach to mirror how medical knowledge is organized and applied in clinical practice.",
  "span_analysis": "Analyzing this text for knowledge structure alignment: The phrase 'I'll approach this using the standard differential diagnosis methodology employed in clinical medicine' shows domain-specific organization by using the exact framework employed in medical practice. The text 'I'll organize the information according to the classic medical history structure:' demonstrates field-appropriate frameworks by following established medical knowledge structures. These demonstrate exceptional knowledge structure alignment through precise medical domain expertise and systematic clinical reasoning patterns.",
  "spans": [[37, 140], [149, 230]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of knowledge structure alignment is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/logical-coherence-trace-prompt.txt
# -----------------------------------------------------------------------------
LOGICAL_COHERENCE_PROMPT = r"""# Annotation Guidelines: Logical Coherence in the Reasoning Process

## Definition
**Logical coherence** is the ability to maintain consistency in reasoning across steps and contexts. In reasoning traces, logical coherence refers to when the participant ensures that conclusions follow validly from premises, maintains internal consistency throughout the reasoning process, and avoids contradictions or invalid inferences.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates logical coherence:

1. **Valid inference chains**: Does the participant draw conclusions that logically follow from premises?
   - Look for proper application of logical rules and inference patterns
   - Check if the participant avoids logical fallacies or unjustified jumps in reasoning

2. **Internal consistency**: Does the participant maintain consistency throughout its reasoning?
   - Look for an absence of contradictory statements or positions
   - Check if later reasoning steps are compatible with earlier statements

3. **Premise tracking**: Does the participant keep track of what has been established and what hasn't?
   - Look for clear connections between premises and conclusions
   - Check if the participant bases its reasoning on properly established facts

4. **Appropriate logical structure**: Does the participant use appropriate logical forms for the context?
   - Look for proper application of deductive, inductive, or abductive reasoning where appropriate
   - Check if the participant structures arguments in logically valid forms

## Label Levels

**0 - Absent**: The reasoning trace shows little to no logical coherence. The participant makes unjustified leaps in reasoning, contradicts itself, or draws conclusions that don't follow from premises.

**1 - Partially Present**: The reasoning trace shows some logical coherence, but with notable gaps or inconsistencies. The participant generally attempts to maintain logical structure but may make occasional invalid inferences or fail to fully connect all reasoning steps.

**2 - Present**: The reasoning trace shows clear logical coherence throughout. The participant draws valid conclusions from premises, maintains internal consistency, properly tracks what has been established, and structures reasoning in logically appropriate forms.

## Span Identification Instructions

In addition to scoring the overall presence of logical coherence, you must identify specific spans (text segments) in the reasoning trace that demonstrate logical coherence. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of logical coherence (valid inference chains, internal consistency, premise tracking, or appropriate logical structure)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to logical coherence
- **Multiple spans**: Identify all significant instances of logical coherence, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of logical coherence are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates logical coherence

### How to Identify Valid Spans
1. **Look for logical coherence language**: Words like "therefore," "thus," "because," "since," "it follows that," "given that," "consequently," "this implies"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant maintaining logical consistency or drawing valid conclusions?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether logical coherence is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate logical coherence. Describe what specific phrases or sentences show valid inference chains, internal consistency, premise tracking, or appropriate logical structure, and explain why they demonstrate logical coherence. This guides you to identify the character indices. Use an empty string "" if no logical coherence is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate logical coherence. Use an empty list [] if no spans demonstrate logical coherence.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "Let's determine if all mammals can fly. Bats are mammals that can fly. Therefore, all mammals can fly. But wait, dogs are mammals and they can't fly. So some mammals can fly and others can't. But birds can fly, and they're animals. So if you're an animal, you might be able to fly. Therefore, some mammals can fly because they're animals."

Annotation:
```json
{{
  "explanation": "The participant demonstrates poor logical coherence by making invalid generalizations, drawing contradictory conclusions, conflating different categories, and presenting a disjointed reasoning chain that doesn't follow valid inference patterns.",
  "span_analysis": "This text shows no logical coherence - it contains invalid generalizations, contradictory statements, and faulty reasoning chains without proper logical structure or valid inference patterns.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To determine whether the triangle is isosceles, I need to check if at least two sides have equal length. The triangle has sides of lengths 5 cm, 5 cm, and 8 cm. Since two sides have equal length (5 cm), this is an isosceles triangle. Isosceles triangles have two equal angles opposite the equal sides. So this triangle has two equal angles. Since angles in a triangle sum to 180°, and two angles are equal, let's call each equal angle x. Then we have x + x + third angle = 180°, so 2x + third angle = 180°. Therefore, each of the equal angles measures (180° - third angle)/2."

Annotation:
```json
{{
  "explanation": "The participant maintains basic logical coherence by correctly identifying the triangle as isosceles based on side lengths and making valid inferences about angle relationships, but doesn't fully establish all connections in its reasoning chain when calculating specific angle values.",
  "span_analysis": "Looking for logical coherence in this text: The phrase 'Since two sides have equal length (5 cm), this is an isosceles triangle' shows valid inference from premise to conclusion. The text 'So this triangle has two equal angles' demonstrates logical deduction from the isosceles property. The phrase 'Since angles in a triangle sum to 180°, and two angles are equal' shows premise tracking and logical structure. The text 'Therefore, each of the equal angles measures (180° - third angle)/2' shows a logical conclusion, though the reasoning chain has some gaps in fully establishing the specific calculation.",
  "spans": [[161, 233], [302, 340], [341, 405], [507, 575]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "Let's examine the validity of the following argument:
Premise 1: All scientists are logical thinkers.
Premise 2: Some artists are logical thinkers.
Conclusion: Some artists are scientists.

I'll analyze this using categorical logic. Let's define our sets:
S = scientists
L = logical thinkers
A = artists

The premises can be symbolized as:
Premise 1: All S are L. (Universal affirmative)
Premise 2: Some A are L. (Particular affirmative)
Conclusion: Some A are S. (Particular affirmative)

The structure of this argument doesn't follow any valid syllogistic form. The middle term (L) is not distributed in either premise, which violates the rules for valid categorical syllogisms. Specifically, this is the fallacy of the undistributed middle.

Let me verify this with a concrete example: Consider logical thinkers as a large set that fully contains scientists. Now imagine some artists fall within the logical thinkers set, but in a different region than the scientists. In this case, both premises would be true, but the conclusion would be false, which proves the argument form is invalid.

Therefore, the given argument is invalid because the conclusion doesn't necessarily follow from the premises, even when the premises are true."

Annotation:
```json
{{
  "explanation": "The participant demonstrates excellent logical coherence by systematically analyzing the argument structure, correctly applying rules of categorical logic, identifying the specific logical fallacy, providing a counterexample that confirms the invalidity, and drawing a conclusion that follows necessarily from the analysis.",
  "span_analysis": "Analyzing this text for logical coherence: The phrase 'I'll analyze this using categorical logic' shows appropriate logical structure by choosing a systematic approach. The text 'The premises can be symbolized as:' demonstrates premise tracking and formal logical structure. The phrase 'The structure of this argument doesn't follow any valid syllogistic form' shows valid inference about argument validity. The text 'which violates the rules for valid categorical syllogisms' demonstrates internal consistency and proper application of logical rules. The phrase 'Let me verify this with a concrete example:' shows systematic verification. The text 'Therefore, the given argument is invalid because the conclusion doesn't necessarily follow from the premises' demonstrates a valid final conclusion. These show excellent logical coherence throughout.",
  "spans": [[190, 232], [305, 339], [490, 563], [622, 680], [745, 788], [1094, 1202]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of logical coherence is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/network-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
NETWORK_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Network Organization in the Reasoning Process

## Definition
**Network organization** is the ability to arrange concepts as interconnected nodes with multiple pathways and relationship types. In reasoning traces, network organization refers to when the participant demonstrates the ability to represent knowledge as a network of interconnected concepts, establish various types of relationships between ideas, and navigate different pathways through this conceptual network.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates network organization:

1. **Network representation**: Does the participant represent information as interconnected nodes?
   - Look for explicit connections between multiple related concepts
   - Check if the participant treats ideas as part of an interconnected network rather than isolated facts

2. **Multiple relationship types**: Does the participant establish different types of connections between concepts?
   - Look for various relationship types (causal, hierarchical, temporal, etc.) between ideas
   - Check if the participant distinguishes between different ways concepts can relate to each other

3. **Multi-pathway exploration**: Does the participant explore multiple paths through the conceptual network?
   - Look for consideration of different routes to conclusions
   - Check if the participant can follow various connection paths between concepts

4. **Network-based inference**: Does the participant make inferences based on the network structure?
   - Look for conclusions drawn from patterns of connections
   - Check if the participant identifies implications from the network topology

## Label Levels

**0 - Absent**: The reasoning trace shows little to no network organization. The participant presents concepts as isolated or in simple linear sequences without establishing a network of relationships or exploring multiple pathways.

**1 - Partially Present**: The reasoning trace shows some network organization, but with limited complexity or inconsistent application. The participant establishes some connections between concepts and may identify different relationship types, but doesn't fully develop or utilize a network structure.

**2 - Present**: The reasoning trace shows clear network organization throughout. The participant consistently represents knowledge as an interconnected network, establishes various relationship types, explores multiple pathways, and leverages the network structure for inference.

## Span Identification Instructions

In addition to scoring the overall presence of network organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate network organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of network organization (network representation, multiple relationship types, multi-pathway exploration, or network-based inference)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to network organization
- **Multiple spans**: Identify all significant instances of network organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of network organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates network organization

### How to Identify Valid Spans
1. **Look for network language**: Words like "connected," "network," "relationships," "pathways," "nodes," "links," "interconnected," "web," "multiple routes," "various connections"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant representing information as interconnected concepts with multiple relationship types?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether network organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate network organization. Describe what specific phrases or sentences show network representation, multiple relationship types, multi-pathway exploration, or network-based inference, and explain why they demonstrate network organization. This guides you to identify the character indices. Use an empty string "" if no network organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate network organization. Use an empty list [] if no spans demonstrate network organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To solve this quadratic equation, I'll follow these steps: First, I need to put it in standard form ax² + bx + c = 0. Then I'll use the quadratic formula x = (-b ± √(b² - 4ac))/2a to find the roots. Substituting the values, I get x = 3 and x = -2. Therefore, the solutions to the equation are x = 3 and x = -2."

Annotation:
```json
{{
  "explanation": "The participant presents a linear, procedural approach with no evidence of representing concepts as an interconnected network, establishing different types of relationships, or exploring multiple pathways.",
  "span_analysis": "This text shows no network organization - it follows a linear procedural sequence without representing concepts as interconnected nodes, establishing multiple relationship types, or exploring different pathways.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Let's analyze how inflation affects different parts of the economy. Inflation directly increases prices, which reduces purchasing power for consumers. This connects to consumer spending, which typically decreases for non-essential goods. Meanwhile, inflation can relate to wages in different ways: sometimes wages increase in response to inflation, but often they lag behind. Inflation also connects to interest rates, as central banks often raise rates to combat high inflation. These higher rates then affect borrowing, housing markets, and business investment. There's also a connection to currency values, as high inflation can lead to currency depreciation."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some network organization by establishing various connections between inflation and other economic concepts, but doesn't fully develop these into a complex network, clearly distinguish relationship types, or explore multiple inference pathways through the connections.",
  "span_analysis": "Looking for network organization in this text: The phrase 'This connects to consumer spending' shows network representation by establishing connections between concepts. The text 'inflation can relate to wages in different ways' demonstrates multiple relationship types by acknowledging different connection patterns. The phrase 'Inflation also connects to interest rates' shows continued network representation. The text 'There's also a connection to currency values' demonstrates network-based thinking by identifying additional connections. These show basic network organization through concept connections.",
  "spans": [[156, 188], [260, 310], [418, 458], [571, 616]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To understand the complex causes and effects of climate change, I'll map out the interconnected factors as a causal network.

Carbon dioxide emissions form a central node in this network, with bidirectional connections to several systems. CO₂ connects to the carbon cycle through absorption relationships with oceans (which absorb ~30% of emissions) and forests (which sequester carbon during growth). However, these same absorption mechanisms create feedback loops: as oceans absorb CO₂, they become more acidic, which can eventually reduce their absorption capacity.

Methane emissions represent another important node, with stronger but shorter-term warming effects than CO₂. Methane connects to agricultural practices (especially rice cultivation and livestock) through production relationships, and to permafrost through a dangerous feedback loop: warming melts permafrost, which releases methane, which causes more warming.

Following another pathway from emissions, we reach atmospheric temperature change, which branches into multiple consequence pathways: one leads to sea level rise (through both thermal expansion and ice sheet melting), another to precipitation pattern changes, and a third to extreme weather events.

The economic system connects to emissions through multiple pathways: energy production (predominantly fossil fuels), industrial processes, transportation systems, and consumption patterns. Each of these represents a different type of relationship - some causal, some correlational, some representing dependency.

If we follow the pathway from climate policies to economic systems, we find both inhibitory relationships (carbon taxes reducing emissions) and synergistic relationships (renewable subsidies promoting technological innovation).

Following yet another path, from precipitation changes to agricultural systems, we discover complex relationships that vary by region - some areas benefit from increased rainfall while others suffer from drought, creating a geonetworkically heterogeneous network of effects.

By mapping these interconnected factors as a complex network rather than linear chains, we can identify crucial intervention points, feedback loops, and emergent properties that might otherwise remain obscure."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated network organization by explicitly representing climate factors as an interconnected network with multiple nodes, identifying diverse relationship types (causal, feedback loops, bidirectional, inhibitory), exploring different pathways through this conceptual network, and drawing insights from the network structure itself.",
  "span_analysis": "Analyzing this text for network organization: The phrase 'I'll map out the interconnected factors as a causal network' explicitly establishes network representation by organizing concepts as interconnected nodes. The text 'Carbon dioxide emissions form a central node in this network, with bidirectional connections to several systems' demonstrates network representation with multiple relationship types. The phrase 'CO₂ connects to the carbon cycle through absorption relationships' shows network-based connections between concepts. These demonstrate sophisticated network organization through explicit network mapping and multiple relationship types.",
  "spans": [[64, 124], [126, 238], [239, 304]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of network organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/ordinal-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
ORDINAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Ordinal Organization in the Reasoning Process

## Definition
**Ordinal organization** is the ability to arrange elements according to relative rank or position. In reasoning traces, ordinal organization refers to when the participant demonstrates the ability to rank items along comparative dimensions, establish preference orders, reason about relative positions, or prioritize information based on importance or relevance.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates ordinal organization:

1. **Comparative ranking**: Does the participant explicitly rank items along one or more dimensions?
   - Look for comparisons using terms like "greater than," "more important than," or "higher priority"
   - Check if the participant orders elements according to specific criteria

2. **Preference ordering**: Does the participant establish clear preference orders?
   - Look for evaluations of options in terms of desirability or optimality
   - Check if the participant creates an ordered list of preferences

3. **Gradation recognition**: Does the participant recognize degrees or gradations rather than binary distinctions?
   - Look for acknowledgment of spectrum positions rather than all-or-nothing categories
   - Check if the participant identifies where items fall along continua

4. **Prioritization**: Does the participant prioritize information or actions based on importance?
   - Look for explicit statements about what matters most or least
   - Check if the participant orders its reasoning according to significance or relevance

## Label Levels

**0 - Absent**: The reasoning trace shows little to no ordinal organization. The participant treats items as unranked, fails to establish comparative relationships, and doesn't prioritize information based on importance.

**1 - Partially Present**: The reasoning trace shows some ordinal organization, but with limited development or inconsistent application. The participant occasionally ranks items or establishes preference orders, but doesn't consistently apply ordinal principles throughout its reasoning.

**2 - Present**: The reasoning trace shows clear ordinal organization throughout. The participant consistently ranks items along comparative dimensions, establishes well-defined preference orders, recognizes gradations, and prioritizes information based on importance.

## Span Identification Instructions

In addition to scoring the overall presence of ordinal organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate ordinal organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of ordinal organization (ranking along dimensions, preference ordering, gradation recognition, or importance prioritization)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to ordinal organization
- **Multiple spans**: Identify all significant instances of ordinal organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of ordinal organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates ordinal organization

### How to Identify Valid Spans
1. **Look for ordinal organization language**: Words like "first," "second," "most important," "less critical," "higher priority," "ranked," "ordered," "better than," "worse than," "more/less"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant organizing information in ranked or ordered relationships?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether ordinal organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate ordinal organization. Describe what specific phrases or sentences show ranking along dimensions, preference ordering, gradation recognition, or importance prioritization, and explain why they demonstrate ordinal organization. This guides you to identify the character indices. Use an empty string "" if no ordinal organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate ordinal organization. Use an empty list [] if no spans demonstrate ordinal organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use carbon dioxide, water, and sunlight to produce glucose and oxygen. The light reactions occur in the thylakoid membrane, while the Calvin cycle takes place in the stroma. Chlorophyll is the primary pigment involved in capturing light energy. Photosynthesis is essential for life on Earth because it produces oxygen and serves as the base of most food chains."

Annotation:
```json
{{
  "explanation": "The participant provides a descriptive explanation of photosynthesis without establishing any ranking, preference ordering, gradation recognition, or prioritization among the elements discussed.",
  "span_analysis": "This text shows no ordinal organization - it presents factual information about photosynthesis without any ranking, ordering, or prioritization of elements.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "When considering which laptop to purchase, you should look at several factors. Processing power is probably the most important feature for your needs as a graphic designer. Memory and storage are also key considerations, with memory being more crucial for your multitasking requirements. Screen quality matters too, especially color accuracy for your work. Battery life is less critical since you mentioned you'll mostly use it plugged in. As for brands, Dell and Apple tend to have better options for creative professionals than Lenovo or HP in this specific use case."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some ordinal organization by ranking certain features as more important than others and establishing a basic preference order among brands, but doesn't develop a comprehensive or systematic ranking system across all dimensions.",
  "span_analysis": "Looking for ordinal organization in this text: The phrase 'Processing power is probably the most important feature' shows importance prioritization. The text 'with memory being more crucial for your multitasking requirements' demonstrates comparative ranking. The phrase 'Battery life is less critical' shows gradation recognition by establishing relative importance. The text 'Dell and Apple tend to have better options for creative professionals than Lenovo or HP' shows preference ordering among brands. These demonstrate basic ordinal organization but without systematic ranking.",
  "spans": [[107, 168], [218, 279], [360, 389], [440, 538]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To evaluate these five treatment options for chronic lower back pain, I'll rank them across multiple dimensions and then establish an overall preference ordering.

First, let's rank them by efficacy based on clinical trial evidence:
1. Physical therapy (highest success rate at 68%)
2. Prescription anti-inflammatories (59% effective)
3. Surgical intervention (52% effective, but with higher variability)
4. Over-the-counter pain relievers (45% effective)
5. Acupuncture (41% effective, but studies show mixed results)

For safety and risk profile, the ranking differs:
1. Over-the-counter pain relievers (lowest serious risk profile)
2. Physical therapy (minimal risks, mainly temporary discomfort)
3. Acupuncture (slightly higher risks of infection or nerve damage)
4. Prescription anti-inflammatories (moderate risks of GI issues and cardiovascular events)
5. Surgical intervention (highest risk of complications)

Considering cost-effectiveness:
1. Over-the-counter pain relievers (lowest cost per QALY gained)
2. Physical therapy (moderate initial cost, high long-term value)
3. Acupuncture (moderate cost, variable effectiveness)
4. Prescription anti-inflammatories (higher cost, especially long-term)
5. Surgical intervention (highest cost, not always proportionate to outcomes)

For long-term sustainability of results:
1. Physical therapy (addresses underlying causes, teaches self-management)
2. Surgical intervention (can provide permanent correction in ideal cases)
3. Acupuncture (benefits often diminish without ongoing treatment)
4. Prescription anti-inflammatories (efficacy often decreases over time)
5. Over-the-counter pain relievers (primarily symptomatic relief)

Weighing these dimensions by relative importance (efficacy: 40%, safety: 30%, cost-effectiveness: 20%, sustainability: 10%), I can establish an overall preference ordering:
1. Physical therapy (scores highest overall due to strong ranking across all dimensions)
2. Over-the-counter pain relievers (strong safety and cost profile compensate for lower efficacy)
3. Prescription anti-inflammatories (balanced profile but with notable safety concerns)
4. Acupuncture (lower risks but also lower proven efficacy)
5. Surgical intervention (despite moderate efficacy, the risk profile and cost make it less preferable as a first-line treatment)

This ordinal analysis suggests a stepped approach, starting with physical therapy, then progressing through the options in order if needed."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated ordinal organization by explicitly ranking treatment options along multiple dimensions with numerical specificity, recognizing gradations within each dimension, weighting the dimensions themselves by relative importance, and synthesizing these rankings into a comprehensive preference ordering with clear reasoning.",
  "span_analysis": "Analyzing this text for ordinal organization: The phrase 'I'll rank them across multiple dimensions and then establish an overall preference ordering' explicitly establishes the ordinal approach. The text 'First, let's rank them by efficacy based on clinical trial evidence:' shows systematic ranking along dimensions. The numbered lists '1. Physical therapy (highest success rate at 68%)' and '2. Prescription anti-inflammatories (59% effective)' demonstrate clear ranking with numerical ordering and gradation recognition. The phrase 'Weighing these dimensions by relative importance (efficacy: 40%, safety: 30%, cost-effectiveness: 20%, sustainability: 10%), I can establish an overall preference ordering:' shows sophisticated importance prioritization and multi-dimensional ranking synthesis. These demonstrate advanced ordinal organization across multiple dimensions with systematic weighting.",
  "spans": [[70, 162], [164, 232], [233, 282], [283, 334], [1123, 1275]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of ordinal organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/pattern-recognition-trace-prompt.txt
# -----------------------------------------------------------------------------
PATTERN_RECOGNITION_PROMPT = r"""# Annotation Guidelines: Pattern Recognition in the Reasoning Process

## Definition
**Pattern recognition** is the ability to recognize recurring structures across different contexts. In reasoning traces, pattern recognition refers to when the participant demonstrates the ability to identify similarities between problems or situations, recognize common patterns in different domains, detect regularities in information, and leverage knowledge of patterns to guide reasoning.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates pattern recognition:

1. **Similarity detection**: Does the participant identify similarities across different cases or instances?
   - Look for recognition of shared features between distinct examples
   - Check if the participant notices commonalities even when surface details differ

2. **Recurring structure identification**: Does the participant recognize repeated patterns or structures?
   - Look for identification of recurring formats, sequences, or relationships
   - Check if the participant notices when the same underlying pattern appears in different contexts

3. **Pattern generalization**: Does the participant extract general patterns from specific instances?
   - Look for abstractions that capture common elements across examples
   - Check if the participant formulates generalizations based on observed regularities

4. **Pattern-based inference**: Does the participant use recognized patterns to guide reasoning?
   - Look for predictions or inferences based on identified patterns
   - Check if the participant leverages pattern knowledge to address new situations

## Label Levels

**0 - Absent**: The reasoning trace shows little to no pattern recognition. The participant treats each case or example as isolated without identifying similarities, recognizing recurring structures, or leveraging patterns.

**1 - Partially Present**: The reasoning trace shows some pattern recognition, but with limited depth or inconsistent application. The participant occasionally identifies similarities or recognizes some patterns, but doesn't consistently detect or leverage patterns throughout its reasoning.

**2 - Present**: The reasoning trace shows clear pattern recognition throughout. The participant consistently identifies similarities across different cases, recognizes recurring structures, extracts general patterns, and effectively uses pattern knowledge to guide reasoning.

## Span Identification Instructions

In addition to scoring the overall presence of pattern recognition, you must identify specific spans (text segments) in the reasoning trace that demonstrate pattern recognition. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of pattern recognition (similarity detection, recurring structure identification, pattern generalization, or pattern-based inference)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to pattern recognition
- **Multiple spans**: Identify all significant instances of pattern recognition, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of pattern recognition are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates pattern recognition

### How to Identify Valid Spans
1. **Look for pattern recognition language**: Words like "similar to," "pattern," "recurring," "like before," "same structure," "this reminds me of," "follows the pattern"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant recognizing patterns, similarities, or recurring structures?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether pattern recognition is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate pattern recognition. Describe what specific phrases or sentences show similarity detection, recurring structure identification, pattern generalization, or pattern-based inference, and explain why they demonstrate pattern recognition. This guides you to identify the character indices. Use an empty string "" if no pattern recognition is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate pattern recognition. Use an empty list [] if no spans demonstrate pattern recognition.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the derivative of f(x) = 3x² + 2x - 5, I'll use the power rule and the sum rule. For the term 3x², the derivative is 3 · 2x = 6x. For the term 2x, the derivative is 2. For the constant term -5, the derivative is 0. Adding these results: f'(x) = 6x + 2 + 0 = 6x + 2. Therefore, the derivative of the function f(x) = 3x² + 2x - 5 is f'(x) = 6x + 2."

Annotation:
```json
{{
  "explanation": "The participant applies calculus rules to find a derivative without demonstrating any pattern recognition, comparison across different cases, or identification of recurring structures; it simply executes a standard procedure on a single example.",
  "span_analysis": "This text shows no pattern recognition - it follows a straightforward calculus procedure without identifying similarities, recurring structures, or patterns across different contexts.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "Looking at the sequence 3, 6, 11, 18, 27, ..., I need to find the next number. First, let me check if this is an arithmetic sequence by looking at the differences between consecutive terms: 6-3=3, 11-6=5, 18-11=7, 27-18=9. These differences are increasing by 2 each time (3, 5, 7, 9), so this isn't an arithmetic sequence.

Let me try a different approach. I notice that the differences form a pattern: 3, 5, 7, 9. These are odd numbers starting from 3. So each term in the original sequence is calculated by adding the next odd number (starting from 3) to the previous term. To find the next number after 27, I need to add the next odd number after 9, which is 11. So 27+11=38. The next number in the sequence is 38."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some pattern recognition by identifying a regular structure in the differences between sequence terms, but this represents a relatively straightforward pattern analysis on a single sequence without comparing across contexts or leveraging patterns for deeper insights.",
  "span_analysis": "Looking for pattern recognition in this text: The phrase 'These differences are increasing by 2 each time (3, 5, 7, 9)' shows recurring structure identification by recognizing a regular pattern in the differences. The text 'I notice that the differences form a pattern: 3, 5, 7, 9' explicitly demonstrates pattern recognition. The phrase 'These are odd numbers starting from 3' shows pattern generalization by identifying the underlying mathematical structure. These demonstrate basic pattern recognition within a single sequence context.",
  "spans": [[223, 283], [357, 414], [415, 453]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "Analyzing these biological invasion case studies, I notice several recurring patterns despite the different species and ecosystems involved.

First, there's a consistent introduction phase pattern across cases. In the zebra mussel, kudzu vine, and cane toad examples, all show an initial lag period where population growth appeared manageable, followed by exponential expansion. This matches the classic "invasion curve" seen in theoretical models. I also notice that each case features an anthropogenic introduction vector—shipping ballast water (zebra mussels), deliberate planting (kudzu), and biocontrol introduction (cane toads).

A second pattern emerges in ecological impact mechanisms. Despite being completely different taxa (mollusk, plant, amphibian), each invader disrupts ecosystems through resource monopolization. The zebra mussels filter excessive water volumes, kudzu blocks light through physical coverage, and cane toads monopolize insect prey. This pattern suggests that successful invaders often share the ability to capture resources at unsustainable rates regardless of their taxonomic group.

I also observe a recurring management failure pattern. In each case, early control efforts focused on adult organisms when populations were already established, rather than targeting introduction pathways or early detection. The initial resource allocations were consistently smaller than later emergency responses by 1-2 orders of magnitude. This pattern suggests a general principle: invasion management tends to be reactive rather than preventative across different government systems and geographies.

The most interesting pattern emerges when examining invasion success factors. All three cases show species with: (1) high reproductive rates, (2) dietary or habitat flexibility, and (3) lack of specialized predators in the invaded range. This "invasion triad" appears consistently despite the cases spanning aquatic, terrestrial, and amphibious habitats across different continents.

Based on these recognized patterns, I can make predictions about the current emerald ash borer outbreak. Despite being a different taxon (insect) in a different ecosystem, I would predict: a rapid acceleration after the current lag phase, resource monopolization through ash phloem consumption, inadequate early management followed by larger emergency funding, and success driven by the invasion triad factors—all of which would suggest similar management failures without intervention that breaks these patterns."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated pattern recognition by identifying multiple recurring structures across diverse biological invasion cases (different species, ecosystems, and locations), extracting generalizable patterns like the \"invasion curve\" and \"invasion triad,\" noting similarities beneath surface differences, and then applying these recognized patterns to make predictions about a new invasion case.",
  "span_analysis": "Analyzing this text for pattern recognition: The phrase 'I notice several recurring patterns despite the different species and ecosystems involved' shows similarity detection across diverse contexts. The text 'there's a consistent introduction phase pattern across cases' demonstrates recurring structure identification. The phrase 'This matches the classic \"invasion curve\" seen in theoretical models' shows pattern generalization by connecting to established patterns. The text 'A second pattern emerges in ecological impact mechanisms' shows systematic pattern identification. The phrase 'The most interesting pattern emerges when examining invasion success factors' demonstrates advanced pattern recognition. The text 'Based on these recognized patterns, I can make predictions' shows pattern-based inference. These demonstrate sophisticated pattern recognition across multiple dimensions and contexts.",
  "spans": [[50, 140], [149, 210], [379, 448], [636, 693], [1623, 1700], [2007, 2065]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of pattern recognition is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/productivity-trace-prompt.txt
# -----------------------------------------------------------------------------
PRODUCTIVITY_PROMPT = r"""# Annotation Guidelines: Productivity in the Reasoning Process

## Definition
**Productivity** is the ability to generate novel combinations using a finite set of elements. In reasoning traces, productivity refers to when the participant demonstrates the ability to create new ideas, solutions, or approaches by recombining existing concepts or rules in original ways rather than just applying fixed patterns.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates productivity:

1. **Novel combinations**: Does the participant combine existing elements in new ways?
   - Look for unexpected or creative combinations of concepts
   - Check if the participant goes beyond standard applications of formulas or procedures

2. **Generative reasoning**: Does the participant generate new possibilities rather than selecting from predefined options?
   - Look for the creation of multiple potential approaches or solutions
   - Check if the participant produces ideas that weren't explicitly provided in the problem

3. **Adaptive application**: Does the participant adapt known principles to new contexts?
   - Look for flexible application of rules or concepts to unfamiliar situations
   - Check if the participant transfers methods from one domain to another

4. **Insight generation**: Does the participant produce insights that go beyond mechanical application of procedures?
   - Look for "aha moments" where the participant makes a creative connection
   - Check if the participant discovers shortcuts or elegant solutions

## Label Levels

**0 - Absent**: The reasoning trace shows little to no productivity. The participant applies standard procedures or fixed patterns without generating novel combinations or approaches.

**1 - Partially Present**: The reasoning trace shows some productivity, but with limited originality or depth. The participant occasionally combines elements in somewhat novel ways or generates basic alternatives, but largely stays within conventional approaches.

**2 - Present**: The reasoning trace shows clear productivity throughout. The participant generates truly novel combinations, creates multiple original approaches, adaptively applies concepts across contexts, and produces genuine insights.

## Span Identification Instructions

In addition to scoring the overall presence of productivity, you must identify specific spans (text segments) in the reasoning trace that demonstrate productivity. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of productivity (novel combinations, generative reasoning, adaptive application, or insight generation)
- **Completeness**: Include enough context to understand the productivity being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to productivity
- **Multiple spans**: Identify all significant instances of productivity, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of productivity are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of productivity it demonstrates (novel combinations, generative reasoning, adaptive application, or insight generation)
- How it contributes to the overall productivity score

### How to Identify Valid Spans
1. **Look for productivity language**: Words like "novel," "creative," "combine," "adapt," "generate," "original," "new approach," "hybrid"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant creating new combinations or generating original approaches?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether productivity is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate productivity. Describe what specific phrases or sentences show novel combinations, generative reasoning, adaptive application, or insight generation, and explain why they demonstrate productivity. This guides you to identify the character indices. Use an empty string "" if no productivity is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate productivity. Use an empty list [] if no spans demonstrate productivity.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the area of a circle with radius 5 cm, I'll use the formula A = πr². Substituting r = 5 cm: A = π(5 cm)² = π(25 cm²) = 25π cm². Using π ≈ 3.14159, we get A ≈ 25 × 3.14159 ≈ 78.54 cm². Therefore, the area of the circle is approximately 78.54 square centimeters."

Annotation:
```json
{{
  "explanation": "The participant applies a standard formula in a straightforward, algorithmic way without generating any novel combinations, alternative approaches, or original insights.",
  "span_analysis": "This text shows no signs of productivity - it's a mechanical application of a standard formula without any novel combinations, creative adaptations, or original approaches.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "I need to estimate how many piano tuners there are in Chicago. Rather than looking up the answer, I'll try to derive it from basic principles. I'll start with Chicago's population, which is roughly 3 million people. Assuming one piano per 50 households, with average household size of 2.5 people, that gives us 3,000,000 ÷ 2.5 ÷ 50 = 24,000 pianos. If each piano needs tuning once per year, and a tuner can tune 5 pianos per day, working 250 days per year, that's 1,250 pianos per tuner per year. So we need about 24,000 ÷ 1,250 = 19.2, or approximately 20 piano tuners in Chicago."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some productivity by breaking down an unusual estimation problem into component factors and combining them in a reasonable way, though it follows a relatively standard Fermi estimation approach without generating truly novel combinations or multiple alternative methods.",
  "span_analysis": "Looking for productivity in this text: The phrase 'Rather than looking up the answer, I'll try to derive it from basic principles' shows some generative reasoning by choosing to create a solution rather than use existing information. The approach of breaking down the problem into population, households, pianos per household, tuning frequency, and tuner capacity demonstrates adaptive application of estimation principles. However, this follows a fairly standard Fermi estimation approach without truly novel combinations.",
  "spans": [[67, 157]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To solve the problem of efficiently packing different-sized objects into the container, I'll try multiple approaches.

First, let's consider a greedy algorithm where we sort objects by volume and insert the largest ones first. This traditional approach works well for uniform objects but may leave inefficient gaps with irregular shapes.

What if we instead think of this as a 3D Tetris problem? We could rotate objects to fit them into complementary spaces, minimizing gaps. But this requires complex spatial reasoning.

Wait - I'm realizing we could borrow a concept from chemistry: molecular packing. What if we model small objects as "solvent molecules" that can fill the gaps between larger "solute molecules"? This suggests a two-phase approach: position large objects first using minimal spacing, then "pour in" smaller objects to fill interstitial spaces.

We could also apply a simulated annealing algorithm, randomly perturbing the configuration and accepting changes that improve packing density with some probability of accepting suboptimal moves to escape local maxima.

Another novel approach would be to use principles from architecture: create "load-bearing structures" with the sturdiest objects, forming arches and vaults that can support weight while creating stable void spaces specifically shaped to accommodate smaller objects.

The most promising approach seems to be a hybrid: use the architectural principles for large, sturdy objects, the molecular packing model for medium and small objects, and simulated annealing to optimize the overall configuration."

Annotation:
```json
{{
  "explanation": "The participant demonstrates exceptional productivity by generating multiple novel approaches that creatively adapt concepts from diverse domains (Tetris, chemistry, architecture), combines these approaches in original ways, and develops a hybrid solution that wasn't obvious from standard packing algorithms.",
  "span_analysis": "Analyzing this text for productivity: The phrase 'What if we instead think of this as a 3D Tetris problem?' shows novel combination by adapting a game concept to packing. The text 'Wait - I'm realizing we could borrow a concept from chemistry: molecular packing' demonstrates adaptive application by transferring chemistry concepts. The phrase 'We could also apply a simulated annealing algorithm' shows novel combination of optimization techniques. The text 'Another novel approach would be to use principles from architecture' explicitly demonstrates novel combination across domains. Finally, 'The most promising approach seems to be a hybrid' shows insight generation by combining multiple approaches into something new. These demonstrate exceptional productivity through creative cross-domain adaptation and novel combinations.",
  "spans": [[339, 395], [522, 603], [865, 916], [1084, 1151], [1351, 1581]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of productivity is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/representational-restructuring-trace-prompt.txt
# -----------------------------------------------------------------------------
REPRESENTATIONAL_RESTRUCTURING_PROMPT = r"""# Annotation Guidelines: Representational Restructuring in the Reasoning Process

## Definition
**Representational restructuring** is the ability to reformulate problems to reveal new insights. In reasoning traces, representational restructuring refers to when the participant demonstrates the ability to change how a problem is represented, view it from different perspectives, transform the problem structure to make it more tractable, or reframe issues to reveal hidden patterns or solutions.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates representational restructuring:

1. **Problem reformulation**: Does the participant reframe or reformulate the problem?
   - Look for alternative ways of describing or representing the same problem
   - Check if the participant transforms the problem into a different but equivalent form

2. **Perspective shifting**: Does the participant view the problem from different angles?
   - Look for multiple perspectives or viewpoints applied to the same issue
   - Check if the participant considers how different framings affect understanding

3. **Representation transformation**: Does the participant convert between different representational formats?
   - Look for shifts between verbal, visual, symbolic, or other representations
   - Check if the participant translates the problem into a more useful form

4. **Insight generation**: Does restructuring lead to new insights or solution paths?
   - Look for "aha moments" that emerge from restructuring
   - Check if reformulation reveals patterns or solutions not visible in the original framing

## Label Levels

**0 - Absent**: The reasoning trace shows little to no representational restructuring. The participant works with the problem as initially presented without exploring alternative representations or perspectives.

**1 - Partially Present**: The reasoning trace shows some representational restructuring, but with limited depth or inconsistent application. The participant occasionally reformulates aspects of the problem or shifts perspective, but doesn't fully leverage restructuring to generate new insights.

**2 - Present**: The reasoning trace shows clear representational restructuring throughout. The participant actively reformulates problems, shifts between multiple perspectives, transforms representations, and generates new insights through restructuring.

## Span Identification Instructions

In addition to scoring the overall presence of representational restructuring, you must identify specific spans (text segments) in the reasoning trace that demonstrate representational restructuring. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of representational restructuring (problem reformulation, perspective shifting, representation transformation, or insight generation)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to representational restructuring
- **Multiple spans**: Identify all significant instances of representational restructuring, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of representational restructuring are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates representational restructuring

### How to Identify Valid Spans
1. **Look for restructuring language**: Words like "reframe," "reformulate," "different perspective," "another way," "transform," "restructure," "view as," "think of it as," "alternatively"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant changing how the problem is represented or viewed?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether representational restructuring is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate representational restructuring. Describe what specific phrases or sentences show problem reformulation, perspective shifting, representation transformation, or insight generation, and explain why they demonstrate representational restructuring. This guides you to identify the character indices. Use an empty string "" if no representational restructuring is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate representational restructuring. Use an empty list [] if no spans demonstrate representational restructuring.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the area of the circle, I need to use the formula A = πr², where r is the radius. The problem states that the diameter is 10 cm, so the radius is half of that, which is 5 cm. Substituting this value into the formula: A = π(5 cm)² = π(25 cm²) = 25π cm². Using π ≈ 3.14159, I get A ≈ 78.54 cm². Therefore, the area of the circle is approximately 78.54 square centimeters."

Annotation:
```json
{{
  "explanation": "The participant applies a standard formula using the given information without reformulating the problem, shifting perspectives, transforming representations, or generating insights through restructuring.",
  "span_analysis": "This text shows no representational restructuring - it follows a straightforward formula application without reformulating the problem, shifting perspectives, or transforming representations.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To determine if we should launch the marketing campaign now or wait until next quarter, I'll consider this from multiple angles. 

From a financial perspective, launching now would cost $50,000 from this quarter's budget, which is already tight. Waiting would allow us to use next quarter's fresh budget.

Looking at it from a competitive angle, our main rival is launching their campaign next month, so waiting might mean our message gets lost in their promotion.

From a product readiness standpoint, some features won't be fully tested until next month, so immediate promotion might create expectations we can't meet.

Considering these factors, I recommend waiting until early next quarter, which balances budget constraints with competitive timing while ensuring product readiness."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some representational restructuring by examining the marketing decision from multiple perspectives (financial, competitive, product readiness), but doesn't fundamentally reformulate the problem, transform between representational formats, or generate significant insights through restructuring.",
  "span_analysis": "Looking for representational restructuring in this text: The phrase 'I'll consider this from multiple angles' shows perspective shifting by explicitly adopting different viewpoints. The text 'From a financial perspective' demonstrates perspective shifting by reframing the problem through a financial lens. The phrase 'Looking at it from a competitive angle' shows continued perspective shifting. The text 'From a product readiness standpoint' demonstrates another perspective shift. These show basic representational restructuring through multiple perspective adoption.",
  "spans": [[92, 139], [141, 181], [309, 351], [471, 513]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "The problem asks us to determine the minimum number of weighings needed to identify a counterfeit coin that's lighter than the others from a set of 12 identical-looking coins. My initial approach would be sequential elimination, testing pairs of coins, but this could require up to 11 weighings in the worst case.

Let me restructure this problem as a information theoretic challenge: each weighing provides a 3-way outcome (left heavier, right heavier, or balanced). With 3 possible outcomes per weighing, n weighings can distinguish between at most 3^n different cases. Since we need to identify 1 counterfeit from 12 coins, we need to distinguish between 12 possibilities. So we need the smallest n where 3^n ≥ 12. Since 3^2 = 9 (too small) and 3^3 = 27 (sufficient), we need at least 3 weighings.

But wait—I can restructure this more efficiently. Instead of viewing it as identifying which coin is counterfeit, let me reframe it as a decision tree problem. If I use the balance scale optimally, each weighing should eliminate roughly 2/3 of the remaining possibilities.

Let me transform this into a different representation: divide the 12 coins into three groups of 4 coins each, labeled A, B, and C. First weighing: compare A vs B. 
- If A is lighter, the counterfeit is in group A
- If B is lighter, the counterfeit is in group B
- If balanced, the counterfeit is in group C

Now I have 4 coins with one counterfeit. For the second weighing, I take 3 of these coins, call them 1, 2, and 3, and split them: compare 1 vs 2.
- If 1 is lighter, coin 1 is counterfeit
- If 2 is lighter, coin 2 is counterfeit
- If balanced, either coin 3 or the unweighed coin 4 is counterfeit

If it's down to coins 3 and 4, a third weighing determines which is counterfeit.

This restructuring reveals that the problem can be solved in exactly 3 weighings in all cases, which is optimal according to our information theory analysis. The key insight from this representational transformation is that we should maximize information gain by creating equal-sized partitions and utilizing the three-way outcome of each weighing."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated representational restructuring by reformulating the counterfeit coin problem multiple ways (from sequential testing to information theory to decision trees), transforming the representation from individual coins to groups and branches, shifting perspective from coin identification to information partitioning, and generating key insights about optimal weighing strategies that were not visible in the original problem formulation.",
  "span_analysis": "Analyzing this text for representational restructuring: The phrase 'Let me restructure this problem as a information theoretic challenge:' explicitly demonstrates problem reformulation by transforming the problem into a different domain. The text 'But wait—I can restructure this more efficiently' shows continued reformulation and insight generation. The phrase 'Instead of viewing it as identifying which coin is counterfeit, let me reframe it as a decision tree problem' demonstrates perspective shifting and representation transformation by changing the fundamental approach. These demonstrate sophisticated representational restructuring through multiple reformulations and perspective shifts that generate new insights.",
  "spans": [[315, 384], [802, 851], [852, 961]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of representational restructuring is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/selective-attention-trace-prompt.txt
# -----------------------------------------------------------------------------
SELECTIVE_ATTENTION_PROMPT = r"""# Annotation Guidelines: Selective Attention in the Reasoning Process

## Definition
**Selective attention** is the ability to focus on relevant information while filtering out distractions. In reasoning traces, selective attention refers to when the participant demonstrates the ability to identify which information is most important for solving a problem, focus on the relevant details, and ignore irrelevant or misleading information.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates selective attention:

1. **Relevance filtering**: Does the participant distinguish between relevant and irrelevant information?
   - Look for explicit identification of which details matter and which don't
   - Check if the participant focuses on information pertinent to the problem

2. **Information prioritization**: Does the participant prioritize more important information over less important details?
   - Look for emphasis on key factors or critical variables
   - Check if the participant allocates more reasoning effort to central issues

3. **Distraction resistance**: Does the participant avoid being sidetracked by irrelevant information?
   - Look for the participant staying focused despite potential distractions
   - Check if the participant filters out misleading or tangential details

4. **Efficient information use**: Does the participant make efficient use of the most important information?
   - Look for reasoning that leverages key insights effectively
   - Check if the participant builds its solution around the most relevant data

## Label Levels

**0 - Absent**: The reasoning trace shows little to no selective attention. The participant treats all information as equally important, fails to distinguish relevant from irrelevant details, or gets distracted by tangential information.

**1 - Partially Present**: The reasoning trace shows some selective attention, but with limited discrimination or inconsistent application. The participant sometimes identifies relevant information or ignores distractions, but doesn't consistently focus on what's most important throughout its reasoning.

**2 - Present**: The reasoning trace shows clear selective attention throughout. The participant consistently identifies which information is most relevant, prioritizes important details, resists distractions, and efficiently uses key information to solve the problem.

## Span Identification Instructions

In addition to scoring the overall presence of selective attention, you must identify specific spans (text segments) in the reasoning trace that demonstrate selective attention. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of selective attention (relevance filtering, information prioritization, distraction resistance, or efficient information use)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to selective attention
- **Multiple spans**: Identify all significant instances of selective attention, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of selective attention are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates selective attention

### How to Identify Valid Spans
1. **Look for selective attention language**: Words like "focus on," "ignore," "irrelevant," "key," "important," "crucial," "dismiss," "filter out," "concentrate on," "prioritize"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant filtering information, prioritizing relevance, or resisting distractions?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether selective attention is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate selective attention. Describe what specific phrases or sentences show relevance filtering, information prioritization, distraction resistance, or efficient information use, and explain why they demonstrate selective attention. This guides you to identify the character indices. Use an empty string "" if no selective attention is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate selective attention. Use an empty list [] if no spans demonstrate selective attention.
4. **"score"**: Your final score as an integer (0, 1, or 2)


## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To analyze this novel, I'll look at various aspects. The author was born in Dublin in 1882 and spent much of his early life there. He later moved to Paris where he wrote most of his famous works. The novel was published in 1922 and contains many references to Irish culture and history. The protagonist walks through the city over the course of a day. There are 18 episodes in the novel, each with a different style. The novel's structure is complex and experimental. Many literary critics consider it a masterpiece of modernist literature. The author also wrote short stories and a play. The novel contains many references to Homer's Odyssey. The language is complex and includes many puns and wordplay."

Annotation:
```json
{{
  "explanation": "The participant includes a series of disconnected facts about the author and novel without distinguishing between relevant and irrelevant information, prioritizing important aspects, or directing attention to the most significant elements for literary analysis.",
  "span_analysis": "This text shows no selective attention - it presents various facts about the author and novel without filtering for relevance, prioritizing important information, or focusing on key aspects for analysis.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this word problem about a mixture of acid solutions, I need to focus on the concentrations and volumes. The problem mentions that there are 10 liters of 20% acid solution and 5 liters of 50% acid solution that will be mixed together. It also mentions the brand names of the acids and their colors, but those details aren't relevant to calculating the final concentration.

I'll calculate the amount of pure acid in each solution. For the first solution: 10 liters × 20% = 2 liters of pure acid. For the second solution: 5 liters × 50% = 2.5 liters of pure acid. The total amount of pure acid is 2 + 2.5 = 4.5 liters. The total volume is 10 + 5 = 15 liters. Therefore, the concentration of the final mixture is 4.5 ÷ 15 = 0.3 or 30%."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some selective attention by explicitly identifying irrelevant details (brand names, colors) and focusing on the relevant variables (concentrations and volumes), but shows a relatively straightforward filtering process without needing to manage complex distractions or prioritize among multiple relevant factors.",
  "span_analysis": "Looking for selective attention in this text: The phrase 'I need to focus on the concentrations and volumes' shows information prioritization by identifying key variables. The text 'but those details aren't relevant to calculating the final concentration' demonstrates relevance filtering by explicitly dismissing irrelevant information. These show basic selective attention through simple relevance filtering and prioritization.",
  "spans": [[82, 133], [267, 336]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To diagnose the root cause of the manufacturing quality issue, I need to analyze the data strategically, focusing on the most informative variables while filtering out noise.

From the extensive data provided, I'll first identify which parameters are most likely to influence product quality based on domain knowledge. The key variables to focus on are temperature, pressure, and raw material composition, as these directly affect the chemical reaction. I'll set aside variables like ambient humidity, operator shift schedules, and storage duration, which are less likely to be causal given the nature of the defect.

Within the temperature data, the critical factor is the peak temperature during the reaction phase, not the pre-heating or cooling phases. Looking specifically at this parameter, I observe a pattern: defective batches consistently had peak temperatures between 182-187°C, while successful batches maintained 175-180°C. This temperature range stands out as highly relevant.

For pressure data, I need to distinguish between normal process variations and significant deviations. The pressure fluctuations between 2.1-2.3 bar appear within normal operating parameters and show no correlation with defects. I can deprioritize this variable to focus on more promising leads.

Examining raw material composition, there's extensive data on 23 different properties. Rather than analyzing all equally, I'll focus on the variables that could chemically influence the identified temperature issue. The catalyst concentration and monomer purity show the strongest correlation with defective batches. Specifically, when catalyst concentration exceeds 0.15% AND monomer purity falls below 98.2%, temperature control issues emerge. This interaction appears critical.

I'll ignore the data on packaging materials entirely, as the defect manifests before the packaging stage. Additionally, the extensive data on ambient factory conditions shows no meaningful pattern related to the defects.

By selectively focusing on the most relevant variables and their interactions while filtering out the considerable noise in this dataset, the root cause becomes clear: a specific combination of high catalyst concentration and lower monomer purity is creating an exothermic reaction that pushes temperatures above the critical threshold, resulting in the observed defects."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated selective attention by explicitly identifying which variables are most relevant to the manufacturing problem, strategically prioritizing specific parameters within broader categories, systematically filtering out numerous irrelevant data points and noise, focusing on interactions between the most critical factors, and efficiently directing attention to the most informative patterns while ignoring tangential information.",
  "span_analysis": "Analyzing this text for selective attention: The phrase 'I need to analyze the data strategically, focusing on the most informative variables while filtering out noise' explicitly establishes selective attention through information prioritization and distraction resistance. The text 'I'll first identify which parameters are most likely to influence product quality' shows relevance filtering by distinguishing important variables. The phrase 'The key variables to focus on are temperature, pressure, and raw material composition' demonstrates information prioritization by identifying crucial factors. The text 'I'll set aside variables like ambient humidity, operator shift schedules, and storage duration' shows distraction resistance by explicitly filtering out irrelevant information. These demonstrate sophisticated selective attention through systematic relevance filtering and strategic focus.",
  "spans": [[63, 174], [210, 291], [319, 404], [454, 548]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of selective attention is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/self-awareness-trace-prompt.txt
# -----------------------------------------------------------------------------
SELF_AWARENESS_PROMPT = r"""# Annotation Guidelines: Self-awareness in the Reasoning Process

## Definition
**Self-awareness** is the ability to assess one's own knowledge state, capabilities, and task solvability. In the context of reasoning traces, self-awareness refers to when the participant demonstrates awareness of what they know and don't know, expresses appropriate uncertainty, and recognizes their limitations.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates self-awareness:

1. **Knowledge state assessment**: Does the participant explicitly acknowledge what they know or don't know?
   - Look for phrases like "I'm not certain about...", "I know that...", "I'm confident that..."
   - Check if the participant identifies gaps in their knowledge or expresses uncertainty appropriately

2. **Capability evaluation**: Does the participant assess whether they can solve the given problem?
   - Look for statements evaluating task difficulty relative to capabilities
   - Check if the participant recognizes when a task might be beyond their abilities or when they need to approach it carefully

3. **Information sufficiency assessment**: Does the participant evaluate whether they have enough information?
   - Look for requests for clarification or additional information when needed
   - Check if the participant identifies when critical information is missing

4. **Uncertainty expression**: Does the participant appropriately qualify their statements with uncertainty?
   - Look for nuanced confidence expressions rather than uniform certainty
   - Check if confidence levels match the complexity or ambiguity of the problem

## Label Levels

**0 - Absent**: The reasoning trace shows no evidence of self-awareness. The participant proceeds with reasoning without acknowledging knowledge limitations, expressing appropriate uncertainty, or recognizing when they lack sufficient information.

**1 - Partially Present**: The reasoning trace shows some evidence of self-awareness, but they are limited or inconsistent. The participant occasionally expresses uncertainty or acknowledges knowledge limitations, but may do so inappropriately or fail to do so when needed.

**2 - Present**: The reasoning trace shows clear evidence of self-awareness throughout. The participant consistently demonstrates awareness of their knowledge state, expresses appropriate levels of uncertainty, recognizes their capabilities, and identifies when they need more information.

## Span Identification Instructions

In addition to scoring the overall presence of self-awareness, you must identify specific spans (text segments) in the reasoning trace that demonstrate self-awareness. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of self-awareness (knowledge state assessment, capability evaluation, information sufficiency assessment, or uncertainty expression)
- **Completeness**: Include enough context to understand the self-awareness being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to self-awareness
- **Multiple spans**: Identify all significant instances of self-awareness, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of self-awareness are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of self-awareness it demonstrates (knowledge state assessment, capability evaluation, information sufficiency assessment, or uncertainty expression)
- How it contributes to the overall self-awareness score

### How to Identify Valid Spans
1. **Look for self-awareness language**: Words like "I know," "I don't know," "I'm uncertain," "my approach," "I realize," "I should note," "I'm not sure"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant reflecting on their own knowledge, reasoning process, or limitations?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether self-awareness is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate self-awareness. Describe what specific phrases or sentences show metacognitive reflection, knowledge assessment, process monitoring, or limitation recognition, and explain why they demonstrate self-awareness. This guides you to identify the character indices. Use an empty string "" if no self-awareness is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate self-awareness. Use an empty list [] if no spans demonstrate self-awareness.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The answer is definitely $342.7$. By using the formula, we multiply the values and get this exact result. This is the correct approach for all problems of this type."

Annotation:
```json
{{
  "explanation": "The participant exhibits no self-awareness, stating their answer with inappropriate certainty, failing to acknowledge any possible limitations in their knowledge or approach.",
  "span_analysis": "This text shows no self-awareness - the participant expresses absolute certainty ('definitely', 'exact result', 'correct approach') without any metacognitive reflection, knowledge assessment, or recognition of limitations.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this problem, I'll try using the quadratic formula. I'm not completely sure if this is the most efficient approach, but it should work for finding the roots. Let me apply the formula: $x = (-b \pm \sqrt{{(b^2 - 4ac)}})/2a$... The answer appears to be $x = 3$ or $x = -2$."

Annotation:
```json
{{
  "explanation": "The participant shows some self-awareness by acknowledging uncertainty about their chosen approach, but doesn't evaluate whether they have sufficient information or assess their confidence in the final answer.",
  "span_analysis": "Looking for self-awareness in this text: The phrase 'I'm not completely sure if this is the most efficient approach' shows process monitoring and uncertainty expression about their method choice. The word 'appears' in 'The answer appears to be' shows some uncertainty about the result. However, the self-awareness is limited and doesn't extend to broader knowledge assessment.",
  "spans": [[60, 148], [230, 280]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "This question asks about the implications of the Jordan-Dickson theorem for quantum computing. I should note that I'm not familiar with a 'Jordan-Dickson theorem' in this context, and it's possible this is either a novel theorem I'm not familiar with or there might be a confusion in the naming. Based on my knowledge of quantum computing, the most relevant theorems are the No-Cloning theorem and the Holevo bound. Without more specific information about what the Jordan-Dickson theorem states, I can't provide a definitive answer about its implications. If you could clarify what this theorem refers to, I'd be better able to address the question."

Annotation:
```json
{{
  "explanation": "The participant demonstrates clear self-awareness by explicitly identifying a gap in their knowledge, explaining what related information they do know, and requesting clarification rather than proceeding with an uncertain answer.",
  "span_analysis": "Analyzing this text for self-awareness: The phrase 'I should note that I'm not familiar with a Jordan-Dickson theorem in this context' shows explicit knowledge state assessment and limitation recognition. The text 'Based on my knowledge of quantum computing' demonstrates knowledge assessment of what they do know. The phrase 'I can't provide a definitive answer about its implications' shows information sufficiency assessment and appropriate uncertainty expression. Finally, 'If you could clarify what this theorem refers to, I'd be better able to address the question' demonstrates metacognitive awareness of what would improve their ability to respond. These show comprehensive self-awareness across multiple dimensions.",
  "spans": [[95, 178], [296, 338], [496, 554], [556, 648]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of the presence of self-awareness in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/self-evaluation-trace-prompt.txt
# -----------------------------------------------------------------------------
SELF_EVALUATION_PROMPT = r"""# Annotation Guidelines: Self-Evaluation in the Reasoning Process

## Definition
**Self-evaluation** is the ability to assess the quality, correctness, efficiency, and progress of one's reasoning and make adjustments as needed. In reasoning traces, self-evaluation refers to when the participant checks its own work, identifies potential errors, evaluates alternative approaches, and makes corrections or improvements to its reasoning.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates self-evaluation:

1. **Verification of steps**: Does the participant check its reasoning steps for correctness?
   - Look for statements like "Let me verify this calculation," "Checking this step," or "To confirm this is correct..."
   - Check if the participant rechecks intermediate results or verifies logical connections

2. **Error detection and correction**: Does the participant identify and fix mistakes in its reasoning?
   - Look for recognition of errors or inconsistencies
   - Check if the participant corrects itself when it notices problems

3. **Alternative evaluation**: Does the participant assess whether its approach is optimal?
   - Look for consideration of whether there are better or more efficient methods
   - Check if the participant evaluates the efficiency or elegance of its solution

4. **Progress assessment**: Does the participant evaluate whether it's making progress toward the goal?
   - Look for reflections on whether the current approach is working
   - Check if the participant recognizes when it's stuck or going in circles

## Label Levels

**0 - Absent**: The reasoning trace shows no evidence of self-evaluation. The participant proceeds without checking its work, detecting errors, considering alternatives, or assessing progress.

**1 - Partially Present**: The reasoning trace shows some evidence of self-evaluation, but it is limited or inconsistent. The participant may occasionally verify steps or detect errors, but doesn't systematically evaluate its reasoning throughout the process.

**2 - Present**: The reasoning trace shows clear evidence of self-evaluation throughout. The participant consistently checks its work, identifies and corrects errors, evaluates alternative approaches, and assesses its progress toward solving the problem.

## Span Identification Instructions

In addition to scoring the overall presence of self-evaluation, you must identify specific spans (text segments) in the reasoning trace that demonstrate self-evaluation. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of self-evaluation (verification of steps, error detection and correction, alternative evaluation, or progress assessment)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to self-evaluation
- **Multiple spans**: Identify all significant instances of self-evaluation, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of self-evaluation are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates self-evaluation

### How to Identify Valid Spans
1. **Look for self-evaluation language**: Words like "let me check," "verify," "I notice an error," "this doesn't seem right," "let me reconsider," "is this the best approach?"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant evaluating, checking, or correcting their own reasoning?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether self-evaluation is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate self-evaluation. Describe what specific phrases or sentences show verification of steps, error detection and correction, alternative evaluation, or progress assessment, and explain why they demonstrate self-evaluation. This guides you to identify the character indices. Use an empty string "" if no self-evaluation is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate self-evaluation. Use an empty list [] if no spans demonstrate self-evaluation.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the derivative of f(x) = x²sin(x), I'll use the product rule. The product rule states that if f(x) = g(x)h(x), then f'(x) = g'(x)h(x) + g(x)h'(x). In this case, g(x) = x² and h(x) = sin(x). So g'(x) = 2x and h'(x) = cos(x). Applying the product rule: f'(x) = 2x·sin(x) + x²·cos(x). Therefore, the derivative of f(x) = x²sin(x) is f'(x) = 2xsin(x) + x²cos(x)."

Annotation:
```json
{{
  "explanation": "The participant simply executes a solution procedure without checking its work, verifying results, or evaluating its approach at any point in the reasoning process.",
  "span_analysis": "This text shows no self-evaluation - it follows a straightforward procedure without any verification of steps, error detection, alternative evaluation, or progress assessment.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve the equation 2x² - 5x - 3 = 0, I'll use the quadratic formula: x = (-b ± √(b² - 4ac))/2a, where a = 2, b = -5, and c = -3. Plugging these values in: x = (5 ± √(25 + 24))/4 = (5 ± √49)/4 = (5 ± 7)/4. So x = (5 + 7)/4 = 12/4 = 3 or x = (5 - 7)/4 = -2/4 = -0.5.

Let me check my answer by substituting these values back into the original equation. For x = 3: 2(3)² - 5(3) - 3 = 2(9) - 15 - 3 = 18 - 15 - 3 = 0. For x = -0.5: 2(-0.5)² - 5(-0.5) - 3 = 2(0.25) + 2.5 - 3 = 0.5 + 2.5 - 3 = 0. Both values check out, so the solutions are x = 3 and x = -0.5."

Annotation:
```json
{{
  "explanation": "The participant shows some self-evaluation by verifying its answers through substitution back into the original equation, but doesn't evaluate its approach or check intermediate steps during the problem-solving process.",
  "span_analysis": "Looking for self-evaluation in this text: The phrase 'Let me check my answer by substituting these values back into the original equation' shows verification of steps by explicitly checking the final results. The text 'Both values check out' demonstrates the completion of the verification process. However, there's no evaluation of the approach or checking of intermediate steps during the solution process.",
  "spans": [[269, 353], [495, 516]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "I need to find the area under the curve f(x) = x² from x = 1 to x = 3. I'll use integration: ∫₁³ x² dx = [x³/3]₁³ = 3³/3 - 1³/3 = 27/3 - 1/3 = 9 - 1/3 = 8.67.

Wait, let me double-check my arithmetic. 27/3 = 9 and 1/3 = 0.33 (rounded), so 9 - 0.33 = 8.67. Actually, let me be more precise: 9 - 1/3 = 9 - 0.333... = 8.666..., which rounds to 8.67.

I should also verify my integration. The antiderivative of x² is indeed x³/3 + C, but I need to make sure I applied the fundamental theorem of calculus correctly. We have ∫₁³ x² dx = [x³/3]₁³ = (3³/3) - (1³/3) = 9 - 1/3 = 8.67. This looks right.

I could also use a different approach to check. Let me try using the geometric interpretation: the area is approximately the sum of rectangles. Using just two rectangles with width 1 at x = 1 and x = 2, we get 1² + 2² = 1 + 4 = 5, which is an underestimate since the function is increasing. Using three narrower rectangles would give a better approximation. This confirms my answer is in the right ballpark, but the exact value from integration is 8.67 square units."

Annotation:
```json
{{
  "explanation": "The participant demonstrates comprehensive self-evaluation by verifying its arithmetic calculations, double-checking the correctness of its integration, and using an alternative approach (rectangular approximation) to confirm that its answer is reasonable.",
  "span_analysis": "Analyzing this text for self-evaluation: The phrase 'Wait, let me double-check my arithmetic' shows verification of steps by explicitly checking calculations. The text 'I should also verify my integration' demonstrates systematic verification of the solution method. The phrase 'I could also use a different approach to check' shows alternative evaluation by considering different methods. The text 'This confirms my answer is in the right ballpark' demonstrates progress assessment by evaluating the reasonableness of the result. These show comprehensive self-evaluation across multiple dimensions.",
  "spans": [[160, 200], [348, 384], [595, 642], [953, 1001]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of self evaluation is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/sequential-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
SEQUENTIAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Sequential Organization in the Reasoning Process

## Definition
**Sequential organization** is the ability to arrange steps in linear order where sequence matters. In reasoning traces, sequential organization refers to when the participant demonstrates the ability to order reasoning steps logically, establish clear dependencies between consecutive steps, and ensure proper progression from premises to conclusions.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates sequential organization:

1. **Logical ordering**: Does the participant arrange reasoning steps in a coherent order?
   - Look for a clear progression from earlier to later steps
   - Check if the participant organizes steps in a logical sequence rather than haphazardly

2. **Step dependencies**: Does the participant establish dependencies between consecutive steps?
   - Look for later steps that build on earlier ones
   - Check if the participant ensures prerequisites are established before using them

3. **Progressive development**: Does the participant develop ideas incrementally through sequential steps?
   - Look for gradual build-up of complexity or understanding
   - Check if the participant shows how each step contributes to the overall progression

4. **Temporal/procedural clarity**: Does the participant clearly indicate the order of operations or reasoning?
   - Look for explicit sequencing indicators (first, next, then, finally)
   - Check if the participant makes clear which steps must precede others

## Label Levels

**0 - Absent**: The reasoning trace shows little to no sequential organization. The participant presents ideas in a disorganized or arbitrary order without clear dependencies between steps.

**1 - Partially Present**: The reasoning trace shows some sequential organization, but with gaps or inconsistent application. The participant generally arranges steps in logical order, but may have some disorganized elements or fail to establish clear dependencies in some cases.

**2 - Present**: The reasoning trace shows clear sequential organization throughout. The participant consistently arranges steps in a logical sequence, establishes clear dependencies between consecutive steps, and develops ideas progressively from beginning to end.

## Span Identification Instructions

In addition to scoring the overall presence of sequential organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate sequential organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of sequential organization (logical step ordering, dependency establishment, progressive development, or temporal sequencing)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to sequential organization
- **Multiple spans**: Identify all significant instances of sequential organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of sequential organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates sequential organization

### How to Identify Valid Spans
1. **Look for sequential organization language**: Words like "first," "next," "then," "after," "before," "following," "subsequently," "finally," "step by step," "in order"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant organizing information or steps in a logical sequence?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether sequential organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate sequential organization. Describe what specific phrases or sentences show logical step ordering, dependency establishment, progressive development, or temporal sequencing, and explain why they demonstrate sequential organization. This guides you to identify the character indices. Use an empty string "" if no sequential organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate sequential organization. Use an empty list [] if no spans demonstrate sequential organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "The Great Depression had several causes including the stock market crash. Many banks failed during this period. The New Deal was Roosevelt's response. Unemployment reached about 25%. The dust bowl made farming difficult. World War II eventually helped end the economic downturn. Herbert Hoover was president when it started. Many people lost their homes and lived in makeshift settlements called Hoovervilles."

Annotation:
```json
{{
  "explanation": "The participant presents a collection of disconnected facts about the Great Depression without any logical ordering, sequential relationships, or progressive development from one idea to the next.",
  "span_analysis": "This text shows no sequential organization - it presents various facts about the Great Depression in a random order without establishing logical sequences, dependencies, or progressive development.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this equation 2x + 3 = 9, I need to isolate x. I'll subtract 3 from both sides to get 2x = 6. Now I need the value of x, so I divide both sides by 2. This gives me x = 3. I should check this answer by substituting it back into the original equation. If x = 3, then 2(3) + 3 = 9, which is 6 + 3 = 9. This is true, so x = 3 is the solution."

Annotation:
```json
{{
  "explanation": "The participant demonstrates basic sequential organization by arranging algebraic steps in a logical order with clear dependencies between steps, but the sequence follows a standard procedural template without requiring sophisticated sequential planning or organization.",
  "span_analysis": "Looking for sequential organization in this text: The phrase 'I'll subtract 3 from both sides to get 2x = 6' shows logical step ordering. The text 'Now I need the value of x, so I divide both sides by 2' demonstrates dependency establishment between consecutive steps. The phrase 'I should check this answer by substituting it back' shows progressive development. These demonstrate basic sequential organization following a standard algebraic procedure.",
  "spans": [[61, 108], [109, 174], [192, 255]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "I'll investigate how the 2008 financial crisis unfolded by tracing the sequential chain of events and their causal relationships.

First, I need to examine the foundational conditions that created vulnerability in the financial system. During the early 2000s, a combination of low interest rates, financial deregulation, and government policies encouraging homeownership created the preconditions for the crisis.

Second, these conditions enabled the housing bubble to form. From approximately 2002-2006, housing prices rose dramatically, fueled by easy credit and the belief that real estate values would continue to increase indefinitely.

Third, this environment fostered the proliferation of subprime mortgages. Lenders extended loans to borrowers with poor credit histories, often with adjustable rates that would increase significantly after an initial period. This step was crucial because it introduced substantial risk into the system that would later trigger wider problems.

Fourth, these high-risk mortgages were transformed through financial engineering. Investment banks purchased these loans, bundled them into mortgage-backed securities (MBS), and sold them to investors worldwide. This securitization process was a necessary precursor to the global spread of the crisis.

Fifth, and building directly on the previous step, these securities were further repackaged into increasingly complex instruments like collateralized debt obligations (CDOs). The complexity obscured the underlying risks, a critical sequential development that prevented proper risk assessment.

Sixth, when interest rates began rising around 2006-2007, the sequential effects began to cascade: adjustable-rate mortgage payments increased, leading to a rising number of defaults, particularly among subprime borrowers who could no longer afford their payments.

Seventh, these defaults triggered a chain reaction: as foreclosures increased, housing prices declined, which led to more homeowners owing more than their homes were worth, leading to more defaults in a self-reinforcing cycle.

Eighth, the value of mortgage-backed securities collapsed, causing massive losses for financial institutions that had invested heavily in these products. This step could only occur after the previous developments had unfolded.

Finally, the crisis spread throughout the global financial system as institutions faced liquidity problems, culminating in the collapse of major firms like Lehman Brothers in September 2008, which triggered a full-blown financial panic and credit freeze.

This sequential analysis reveals how each development built upon previous ones, creating a chain of events where the specific ordering was crucial to understanding how the crisis unfolded."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated sequential organization by meticulously arranging events in chronological order, explicitly numbering steps to indicate their sequence, establishing clear causal dependencies between consecutive developments, and showing how each step necessarily built upon previous ones in a complex chain of financial events.",
  "span_analysis": "Analyzing this text for sequential organization: The opening 'I'll investigate how the 2008 financial crisis unfolded by tracing the sequential chain of events' explicitly establishes sequential organization as the approach. The numbered sequence markers 'First, I need to examine the foundational conditions', 'Second, these conditions enabled the housing bubble to form', and 'Third, this environment fostered the proliferation of subprime mortgages' demonstrate systematic logical step ordering. The phrase 'Finally, the crisis spread throughout the global financial system' shows temporal sequencing and progressive development. These demonstrate sophisticated sequential organization with explicit numbering and causal dependencies.",
  "spans": [[0, 97], [131, 183], [414, 474], [642, 715], [2107, 2165]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of sequential organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/spatial-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
SPATIAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Spatial Organization in the Reasoning Process

## Definition
**Spatial organization** is the ability to arrange elements according to spatial relationships and configurations. In reasoning traces, spatial organization refers to when the participant demonstrates the ability to reason about relative positions, distances, or arrangements in space, understand how elements relate to each other spatially, and manipulate or transform spatial configurations.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates spatial organization:

1. **Spatial representation**: Does the participant represent spatial relationships between elements?
   - Look for descriptions of where things are located relative to each other
   - Check if the participant creates mental maps or spatial configurations

2. **Relational reasoning**: Does the participant reason about how elements are arranged in space?
   - Look for inferences based on spatial relationships (above, below, adjacent, etc.)
   - Check if the participant uses spatial information to draw conclusions

3. **Spatial transformation**: Does the participant mentally manipulate spatial configurations?
   - Look for reasoning about how arrangements would change after operations
   - Check if the participant can predict the results of spatial transformations

4. **Spatial problem-solving**: Does the participant use spatial strategies to solve problems?
   - Look for spatial approaches to reasoning tasks
   - Check if the participant leverages spatial thinking even for abstract problems

## Label Levels

**0 - Absent**: The reasoning trace shows little to no spatial organization. The participant doesn't represent spatial relationships, reason about arrangements in space, or use spatial strategies for problem-solving.

**1 - Partially Present**: The reasoning trace shows some spatial organization, but with limited sophistication or inconsistent application. The participant sometimes represents spatial relationships or uses basic spatial reasoning, but doesn't consistently employ spatial organization throughout its reasoning.

**2 - Present**: The reasoning trace shows clear spatial organization throughout. The participant consistently represents spatial relationships, reasons effectively about arrangements in space, mentally manipulates spatial configurations, and employs spatial strategies for problem-solving.

## Span Identification Instructions

In addition to scoring the overall presence of spatial organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate spatial organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of spatial organization (spatial relationship representation, spatial reasoning, mental manipulation, or spatial problem-solving strategies)
- **Completeness**: Include complete sentences or meaningful phrases that can be understood in isolation
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to spatial organization
- **Multiple spans**: Identify all significant instances of spatial organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of spatial organization are demonstrated
- **Verification**: After identifying spans, manually verify that the extracted text actually demonstrates spatial organization

### How to Identify Valid Spans
1. **Look for spatial organization language**: Words like "above," "below," "left," "right," "adjacent," "perpendicular," "parallel," "rotate," "flip," "arrange," "position," "layout," "configuration"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant reasoning about spatial relationships, arrangements, or transformations?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether spatial organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate spatial organization. Describe what specific phrases or sentences show spatial relationship representation, spatial reasoning, mental manipulation, or spatial problem-solving strategies, and explain why they demonstrate spatial organization. This guides you to identify the character indices. Use an empty string "" if no spatial organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate spatial organization. Use an empty list [] if no spans demonstrate spatial organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To determine the theme of the poem, I'll analyze its language, imagery, and structure. The poem uses repeated references to the passage of time and mortality. It employs metaphors comparing life to a journey and death to a destination. The tone shifts from contemplative in the first stanza to more accepting in the final stanza. Based on these elements, the central theme appears to be the inevitability of death and the importance of accepting one's mortality."

Annotation:
```json
{{
  "explanation": "The participant conducts a thematic analysis without employing any spatial reasoning, representing spatial relationships, or using spatial organization strategies in its approach.",
  "span_analysis": "This text shows no spatial organization - it analyzes literary themes without any spatial relationship representation, spatial reasoning, or spatial problem-solving strategies.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this geometry problem, I need to find the area of the triangle. The triangle has coordinates at (0,0), (3,0), and (3,4). I can visualize this triangle as having its base along the x-axis from the origin to the point (3,0), with a height extending up to the point (3,4). The base length is 3 units. The height is 4 units. Using the formula Area = (1/2) × base × height, I get Area = (1/2) × 3 × 4 = 6 square units."

Annotation:
```json
{{
  "explanation": "The participant demonstrates basic spatial organization by visualizing the triangle's position on a coordinate plane and identifying its spatial properties, but employs a relatively straightforward spatial approach without sophisticated transformation or complex spatial reasoning.",
  "span_analysis": "Looking for spatial organization in this text: The phrase 'I can visualize this triangle as having its base along the x-axis from the origin to the point (3,0)' shows spatial relationship representation by describing position and orientation. The text 'with a height extending up to the point (3,4)' demonstrates spatial reasoning about vertical relationships. These show basic spatial organization through coordinate visualization and spatial property identification.",
  "spans": [[117, 212], [214, 265]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To solve this puzzle about fitting the furniture in the living room, I'll need to reason carefully about spatial arrangements.

First, let me establish the room's configuration: it's rectangular, 15 feet by 20 feet, with the entrance door on the south wall, windows on the east wall, and a fireplace centered on the north wall. This creates essential constraints that will affect all further arrangements.

The sectional sofa has an L-shape with dimensions 10 feet along one side and a 8 feet along the other. Given these measurements, I need to determine its optimal placement. If we position it with its back to the west wall and extending along the south wall, it would block the entrance door. If we rotate it to have its back to the south wall extending toward the west, the longer section would protrude too far into the room, disrupting the natural pathway from the entrance to the other areas.

The most effective arrangement would place the sofa with its back to the west wall, extending toward the north, with the shorter section of the L facing east. This creates a natural division of the space while maintaining a clear pathway from the entrance. In this configuration, the sofa would be positioned approximately 5 feet from the south wall, leaving adequate clearance for the entrance.

The coffee table (3 feet by 4 feet) should be placed in front of the sofa, but not directly in the walking path. Given the sofa's L-shape, the table would fit best if positioned toward the inner corner of the L, approximately 1.5 feet from the sofa on both sides. This maintains accessibility while creating a functional arrangement.

For the two armchairs (each 3 feet by 3 feet), we have several options. If we place them opposite the sofa, facing west, they would create a conversation area around the coffee table. However, this might crowd the pathway to the fireplace. Alternatively, we could position them at angles in the northeast and southeast corners of the conversation area, which would maintain the pathway while still allowing all seats to face each other.

The entertainment center (6 feet wide) would logically be placed against the east wall, between the windows. In this position, it would be visible from the sofa and armchairs without becoming the room's focal point, which should remain the fireplace.

If we mentally rearrange this configuration, rotating the entire setup 90 degrees counterclockwise, the sofa would block the windows and create an awkward relationship with the fireplace. Similarly, a 90-degree clockwise rotation would place the entertainment center against the fireplace wall, creating a competing focal point.

Therefore, the optimal spatial organization places the sofa against the west wall, the entertainment center against the east wall, and the armchairs completing the conversation area while maintaining clear pathways throughout the space."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated spatial organization by mentally representing a complex room layout, reasoning about relative positions and clearances between multiple objects, simulating different spatial arrangements and transformations (rotations), considering pathways through the space, and using spatial constraints to solve the furniture arrangement problem.",
  "span_analysis": "Analyzing this text for spatial organization: The phrase 'I'll need to reason carefully about spatial arrangements' explicitly establishes spatial reasoning as the approach. The text 'First, let me establish the room's configuration: it's rectangular, 15 feet by 20 feet' shows spatial relationship representation by describing dimensions and layout. The phrase 'with the entrance door on the south wall, windows on the east wall, and a fireplace centered on the north wall' demonstrates detailed spatial positioning using directional references. The text 'If we position it with its back to the west wall and extending along the south wall' shows mental manipulation of spatial configurations. These demonstrate sophisticated spatial organization through detailed spatial reasoning and mental manipulation.",
  "spans": [[69, 126], [128, 214], [216, 327], [579, 662]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of spatial organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/strategy-selection-trace-prompt.txt
# -----------------------------------------------------------------------------
STRATEGY_SELECTION_PROMPT = r"""# Annotation Guidelines: Strategy Selection in the Reasoning Process

## Definition
**Strategy selection** is the ability to choose the most appropriate reasoning approaches based on task requirements and domain. In reasoning traces, strategy selection refers to when the participant demonstrates deliberate choice between different reasoning methods and adapts their approach to the specific problem type.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates strategy selection:

1. **Explicit strategy consideration**: Does the participant explicitly consider which approach to use?
   - Look for statements like "For this type of problem, I'll use...", "I'll approach this by...", or "There are several ways to solve this..."
   - Check if the participant weighs different possible approaches before proceeding

2. **Task-appropriate strategies**: Does the participant select strategies that fit the specific task requirements?
   - Look for reasoning approaches that are well-suited to the particular problem type
   - Check if the strategy matches domain conventions (e.g., mathematical problems use relevant theorems, logical problems use formal logic)

3. **Strategy adaptation**: Does the participant adapt their strategy based on problem features?
   - Look for different approaches used for different aspects of the problem
   - Check if the participant modifies their approach when faced with unique features of the problem

## Label Levels

**0 - Absent**: The reasoning trace shows no evidence of strategy selection. the participant applies a generic or inappropriate approach without considering which strategy would be most effective for the specific task.

**1 - Partially Present**: The reasoning trace shows some evidence of strategy selection, but it is limited or not fully deliberate. The participant may briefly consider which approach to take or use task-appropriate methods, but doesn't thoroughly evaluate strategic options or show strong awareness that they are responsible for choosing the optimal strategy.

**2 - Present**: The reasoning trace shows clear evidence of strategy selection throughout. the participant deliberately selects appropriate strategies based on task characteristics, adapts their approach as needed, and may consider alternative strategies when appropriate.

## Span Identification Instructions

In addition to scoring the overall presence of strategy selection, you must identify specific spans (text segments) in the reasoning trace that demonstrate strategy selection. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of strategy selection
- **Completeness**: Include enough context to understand the strategy selection being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to strategy selection
- **Multiple spans**: Identify all significant instances of strategy selection, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of strategy selection are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of strategy selection it demonstrates (explicit strategy consideration, task-appropriate strategies, or strategy adaptation)
- How it contributes to the overall strategy selection score

### How to Identify Valid Spans
1. **Look for strategy language**: Words like "approach," "method," "strategy," "technique," "way," "instead," "alternatively," "I could," "better approach"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant choosing between different approaches or methods?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether strategy selection is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate strategy selection. Describe what specific phrases or sentences show strategy identification, strategy comparison, strategy evaluation, or strategy choice, and explain why they demonstrate strategy selection. This guides you to identify the character indices. Use an empty string "" if no strategy selection is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate strategy selection. Use an empty list [] if no spans demonstrate strategy selection.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To answer this question, I'll think step by step. First, I need to understand what the question is asking. The problem asks for the value of x in the equation 2x + 3 = 7. I'll subtract 3 from both sides to get 2x = 4. Then I'll divide both sides by 2 to get x = 2. Therefore, x = 2 is the answer."

Annotation:
```json
{{
  "explanation": "The participant applies a standard algebraic approach without showing any deliberate consideration of strategy selection or evaluation of alternative approaches for this problem.",
  "span_analysis": "This text shows no strategy selection - it follows a single, straightforward algebraic approach without considering alternatives, comparing methods, or making strategic choices.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "This is a probability problem involving dependent events. I'll use Bayes' theorem since we're dealing with conditional probabilities. Let's apply Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)..."

Annotation:
```json
{{
  "explanation": "The participant selects the strategy of Bayes' theorem because they realize the problem domain is probability, but doesn't thoroughly explain the strategic choice or considering alternative approaches.",
  "span_analysis": "Looking for strategy selection in this text: The phrase 'I'll use Bayes' theorem since we're dealing with conditional probabilities' shows strategy identification and selection based on problem characteristics. However, there's no consideration of alternative approaches or comparison of methods.",
  "spans": [[0, 132]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "This optimization problem could be approached in several ways. I could use calculus to find critical points, employ a graphical method, or use linear programming techniques. Since we're dealing with a function with multiple variables and constraints, I'll use Lagrange multipliers as the most efficient approach. However, I should first check if the constraints create a convex region to ensure we can find a global optimum rather than just a local one. Looking at the constraints, I can confirm they form a convex set, so Lagrange multipliers will work well here. If we encounter difficulties with this approach, I can switch to examining boundary conditions separately... [later in the solution] Since the Lagrangian isn't yielding a straightforward solution due to the complexity of the resulting equations, I'll switch to a numerical approach by examining critical points at the boundaries of our feasible region..."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated strategy selection by explicitly considering multiple possible approaches, justifying their initial strategy choice based on problem characteristics, verifying preconditions for the selected method, and adaptively switching strategies when the initial approach proves challenging.",
  "span_analysis": "Analyzing this text for strategy selection: The opening 'This optimization problem could be approached in several ways. I could use calculus to find critical points, employ a graphical method, or use linear programming techniques' shows explicit strategy identification and comparison of multiple approaches. The text 'Since the Lagrangian isn't yielding a straightforward solution due to the complexity of the resulting equations, I'll switch to a numerical approach' demonstrates actual strategy switching when the initial approach proves challenging. These show comprehensive strategy selection through both initial comparison and adaptive switching.",
  "spans": [[0, 173], [698, 846]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of strategy selection is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/temporal-organization-trace-prompt.txt
# -----------------------------------------------------------------------------
TEMPORAL_ORGANIZATION_PROMPT = r"""# Annotation Guidelines: Temporal Organization in the Reasoning Process

## Definition
**Temporal organization** is the ability to arrange elements along a timeline with before/after relationships. In reasoning traces, temporal organization refers to when the participant demonstrates the ability to reason about sequences of events, understand how states change over time, establish temporal relationships between occurrences, and consider how past, present, and future states relate to each other.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates temporal organization:

1. **Temporal sequencing**: Does the participant arrange events in chronological order?
   - Look for clear ordering of events along a timeline
   - Check if the participant establishes what happened before or after what

2. **Temporal relationships**: Does the participant reason about relationships in time?
   - Look for reasoning about durations, intervals, or temporal distances
   - Check if the participant uses temporal relationships to draw inferences

3. **State changes over time**: Does the participant track how situations evolve over time?
   - Look for representation of how states transition to other states
   - Check if the participant reasons about processes of change

4. **Temporal contexts**: Does the participant consider how time frames affect interpretation?
   - Look for recognition of how past and future contexts matter
   - Check if the participant adapts reasoning based on temporal perspective

## Label Levels

**0 - Absent**: The reasoning trace shows little to no temporal organization. The participant doesn't arrange events chronologically, reason about temporal relationships, track changes over time, or consider temporal contexts.

**1 - Partially Present**: The reasoning trace shows some temporal organization, but with limited sophistication or inconsistent application. The participant sometimes arranges events chronologically or reasons about temporal relationships, but doesn't consistently employ temporal organization throughout its reasoning.

**2 - Present**: The reasoning trace shows clear temporal organization throughout. The participant consistently arranges events chronologically, reasons effectively about temporal relationships, tracks changes over time, and considers how temporal contexts affect interpretation.

## Span Identification Instructions

In addition to scoring the overall presence of temporal organization, you must identify specific spans (text segments) in the reasoning trace that demonstrate temporal organization. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of temporal organization (temporal sequencing, temporal relationships, state changes over time, or temporal contexts)
- **Completeness**: Include enough context to understand the temporal organization being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to temporal organization
- **Multiple spans**: Identify all significant instances of temporal organization, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of temporal organization are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of temporal organization it demonstrates (temporal sequencing, temporal relationships, state changes over time, or temporal contexts)
- How it contributes to the overall temporal organization score

### How to Identify Valid Spans
1. **Look for temporal language**: Words like "sequence," "timeline," "before/after," "temporal," "chronological," "over time," "then," "next," "simultaneously"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant organizing information by time or tracking temporal relationships?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether temporal organization is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate temporal organization. Describe what specific phrases or sentences show temporal sequencing, chronological tracking, temporal relationships, or time-based structuring, and explain why they demonstrate temporal organization. This guides you to identify the character indices. Use an empty string "" if no temporal organization is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate temporal organization. Use an empty list [] if no spans demonstrate temporal organization.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To calculate the compound interest, we use the formula A = P(1 + r)^t. In this problem, P is the principal amount of $1000, r is the annual interest rate of 5% or 0.05, and t is the time period of 3 years. Substituting these values: A = $1000(1 + 0.05)^3 = $1000(1.05)^3 = $1000 × 1.157625 = $1157.63. Therefore, the total amount after 3 years will be $1157.63, and the compound interest earned is $1157.63 - $1000 = $157.63."

Annotation:
```json
{{
  "explanation": "The participant performs a mathematical calculation that involves time as a variable, but doesn't demonstrate any temporal organization in its reasoning process, as it doesn't arrange events chronologically, reason about temporal relationships, or track changes across time.",
  "span_analysis": "This text shows no temporal organization - while time is mentioned as a variable (3 years), there is no temporal sequencing, chronological tracking, or organization of information by time relationships.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "The process of mitosis has several distinct phases. It begins with prophase, where the chromatin condenses into chromosomes and the nuclear envelope breaks down. Next comes metaphase, where chromosomes align at the cell's equator. This is followed by anaphase, when the sister chromatids separate and move toward opposite poles of the cell. Finally, in telophase, new nuclear membranes form around the two sets of chromosomes, and the cytoplasm divides in cytokinesis, resulting in two daughter cells. The entire process ensures that each daughter cell receives an identical set of chromosomes."

Annotation:
```json
{{
  "explanation": "The participant demonstrates basic temporal organization by sequencing the phases of mitosis in chronological order with clear before/after relationships, but provides a relatively straightforward description of a standard biological process without sophisticated temporal reasoning about intervals, changing states, or temporal contexts.",
  "span_analysis": "Looking for temporal organization in this text: The phrase 'It begins with prophase' shows temporal sequencing by establishing the starting point. The words 'Next comes metaphase' demonstrate chronological tracking with explicit temporal transitions. The phrase 'This is followed by anaphase' continues the temporal sequence. Finally, 'Finally, in telophase' completes the temporal organization. These show basic temporal sequencing but without complex temporal relationships.",
  "spans": [[55, 157], [158, 228], [229, 340], [341, 410]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "To understand how the current Middle East conflict evolved, I need to analyze the temporal patterns and sequences of events that led to this point, while recognizing how different time scales interact.

In the very long-term context (over centuries), the region experienced the decline of the Ottoman Empire, followed by the colonial period of British and French mandates, which established borders that often didn't align with ethnic or religious distributions. This created a foundational instability that continues to influence events today, showing how distant temporal states can have persistent effects.

Moving to the mid-term timescale (decades), several key events occurred in sequence, each building on previous temporal states: the 1948 establishment of Israel, followed by the 1967 Six-Day War, which significantly altered territorial control. Then the 1973 Yom Kippur War changed regional dynamics again, followed by the 1979 Iranian Revolution, which transformed religious-political power structures. The temporal proximity of these events matters—had they occurred with greater time intervals between them, regional adaptations might have developed differently.

In the short-term context (years), we must track how the Arab Spring of 2011 triggered a sequence of rapidly evolving state changes across the region. Initially promising democratic reforms, this process instead led to power vacuums in several countries. The temporal overlap of these simultaneous transitions created unique interaction effects that wouldn't have occurred if they had happened sequentially instead of concurrently.

Most recently (months to weeks), we need to examine how specific triggering events created cascade effects with compressed temporal dynamics. Initial protests escalated to limited confrontations, then to open conflict within just 14 days—a temporal acceleration pattern typical of modern conflicts. This phase requires almost hourly tracking to understand how each action created reactions in increasingly shortened time intervals.

What's particularly important is recognizing how these different temporal scales interact: long-term historical grievances were activated by medium-term political developments, which created conditions for short-term escalations, all of which operate simultaneously in the present moment. Understanding current dynamics requires shifting between these temporal frames.

Looking forward temporally, I can project several potential future states based on historical patterns of how similar conflicts evolved over time. If we examine previous regional conflicts, we observe that those with international involvement typically lasted 3-7 years before reaching stable resolution, suggesting a likely timeline for the current situation.

This temporal organization reveals that what might appear as disconnected events actually form coherent temporal patterns when properly sequenced and analyzed across different time scales."

Annotation:
```json
{{
  "explanation": "The participant demonstrates sophisticated temporal organization by analyzing events across multiple time scales (centuries, decades, years, weeks), explicitly tracking how states change over time, establishing complex temporal relationships between events, considering how different temporal contexts affect interpretations, and even projecting future temporal patterns based on historical sequences.",
  "span_analysis": "Analyzing this text for temporal organization: The phrase 'I need to analyze the temporal patterns and sequences of events' explicitly establishes temporal organization as the approach. The text 'In the very long-term context (over centuries)' demonstrates multi-scale temporal structuring. The phrase 'Moving to the mid-term timescale (decades)' shows explicit temporal sequencing across different time scales. The text 'In the short-term context (years)' continues the multi-temporal framework. The phrase 'Most recently (months to weeks)' demonstrates fine-grained temporal tracking. The text 'What's particularly important is recognizing how these different temporal scales interact' shows sophisticated temporal relationship analysis. Finally, 'Looking forward temporally, I can project several potential future states' demonstrates temporal projection. These show advanced temporal organization across multiple scales and relationships.",
  "spans": [[60, 123], [203, 249], [611, 653], [1178, 1211], [1611, 1642], [2044, 2183], [2414, 2517]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of temporal organization is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# -----------------------------------------------------------------------------
# FILE: element_annotation/verification-trace-prompt.txt
# -----------------------------------------------------------------------------
VERIFICATION_PROMPT = r"""# Annotation Guidelines: Verification in the Reasoning Process

## Definition
**Verification** is the ability to check reasoning steps against established criteria. In reasoning traces, verification refers to when the participant demonstrates the ability to confirm the correctness or validity of its reasoning, evaluate whether conclusions follow from premises, check for errors or inconsistencies, and assess whether reasoning meets relevant standards.

## What to Look For
When analyzing a reasoning trace, look for evidence that the participant demonstrates verification:

1. **Validity checking**: Does the participant verify that conclusions logically follow from premises?
   - Look for assessment of whether inferences are valid
   - Check if the participant confirms that deductions are properly drawn

2. **Error detection**: Does the participant check for mistakes or inconsistencies?
   - Look for identification of potential errors in reasoning
   - Check if the participant catches contradictions or flawed logic

3. **Standard conformity**: Does the participant evaluate reasoning against established criteria?
   - Look for assessment of whether reasoning meets domain-specific standards
   - Check if the participant confirms that appropriate principles or methods are followed

4. **Confirmation procedures**: Does the participant use systematic approaches to verify results?
   - Look for verification techniques like checking examples, testing edge cases, or using alternative methods
   - Check if the participant employs confirmation strategies to validate conclusions

## Label Levels

**0 - Absent**: The reasoning trace shows little to no verification. The participant proceeds without checking validity, detecting errors, evaluating against standards, or confirming results.

**1 - Partially Present**: The reasoning trace shows some verification, but with limited depth or inconsistent application. The participant occasionally checks validity or looks for errors, but doesn't systematically verify reasoning throughout the process.

**2 - Present**: The reasoning trace shows clear verification throughout. The participant consistently checks validity, detects potential errors, evaluates against established standards, and uses confirmation procedures to verify results.

## Span Identification Instructions

In addition to scoring the overall presence of verification, you must identify specific spans (text segments) in the reasoning trace that demonstrate verification. Follow these guidelines:

### Span Format
- Use character indices (0-based) to identify exact text segments
- Format: [start_index, end_index] where reasoning_trace[start_index:end_index] extracts the exact span
- Ensure spans capture complete thoughts or sentences for meaningful analysis
- Character indices should include any leading/trailing spaces that are part of the meaningful text

### Span Selection Criteria
- **Relevance**: The span must clearly demonstrate one or more aspects of verification (validity checking, error detection, standard conformity, or confirmation procedures)
- **Completeness**: Include enough context to understand the verification being demonstrated
- **Precision**: Don't include unnecessary surrounding text that doesn't contribute to verification
- **Multiple spans**: Identify all significant instances of verification, even if they occur in different parts of the response
- **Non-overlapping**: Prefer non-overlapping spans when possible, but overlaps are acceptable if different aspects of verification are demonstrated

### Span Justification
For each identified span, provide a brief explanation of:
- Which type of verification it demonstrates (validity checking, error detection, standard conformity, or confirmation procedures)
- How it contributes to the overall verification score

### How to Identify Valid Spans
1. **Look for verification language**: Words like "verify," "check," "confirm," "validate," "test," "examine," "assess," "evaluate against standards"
2. **Check completeness**: Make sure your span contains complete sentences that make sense alone
3. **Verify the capability**: Ask "Does this text clearly show the participant checking, validating, or confirming their reasoning?"

## Output Format
Provide your analysis in strict JSON format with exactly four fields:

1. **"explanation"**: A brief one-sentence explanation of your reasoning about whether verification is present in the reasoning trace
2. **"span_analysis"**: Your reasoning process for identifying parts of the text that demonstrate verification. Describe what specific phrases or sentences show validity checking, error detection, standard conformity, or confirmation procedures, and explain why they demonstrate verification. This guides you to identify the character indices. Use an empty string "" if no verification is found.
3. **"spans"**: A list of lists, where each sublist contains [start_index, end_index] for character indices that demonstrate verification. Use an empty list [] if no spans demonstrate verification.
4. **"score"**: Your final score as an integer (0, 1, or 2)

## Example Annotations

### Example 1 (Score: 0)
Reasoning trace: "To find the area of the triangle, I'll use the formula A = (1/2)bh, where b is the base and h is the height. The triangle has a base of 6 inches and a height of 4 inches. Substituting these values into the formula: A = (1/2)(6 inches)(4 inches) = (1/2)(24 square inches) = 12 square inches. Therefore, the area of the triangle is 12 square inches."

Annotation:
```json
{{
  "explanation": "The participant simply applies a formula and calculates a result without any attempt to verify the validity of its reasoning, check for errors, confirm the result, or evaluate against standards.",
  "span_analysis": "This text shows no verification - it's a straightforward calculation without any checking, validation, confirmation, or assessment of the reasoning process or result.",
  "spans": [],
  "score": 0
}}
```

### Example 2 (Score: 1)
Reasoning trace: "To solve this equation 3x + 7 = 22, I'll isolate the variable x. First, I'll subtract 7 from both sides: 3x + 7 - 7 = 22 - 7, which simplifies to 3x = 15. Next, I'll divide both sides by 3: 3x/3 = 15/3, giving me x = 5.

Let me verify this answer by substituting x = 5 back into the original equation: 3(5) + 7 = 15 + 7 = 22. Since I get 22, which matches the right side of the original equation, my answer x = 5 is correct."

Annotation:
```json
{{
  "explanation": "The participant demonstrates some verification by checking its answer through substitution back into the original equation, but this represents a single basic verification step rather than a systematic approach to verification throughout the reasoning process.",
  "span_analysis": "Looking for verification in this text: The phrase 'Let me verify this answer by substituting x = 5 back into the original equation' explicitly shows verification through confirmation procedures. The text demonstrates checking the result against the original equation and confirming the answer is correct. However, this is a single, basic verification step rather than systematic verification throughout.",
  "spans": [[196, 396]],
  "score": 1
}}
```

### Example 3 (Score: 2)
Reasoning trace: "I'll analyze whether the study's conclusion that 'coffee consumption reduces risk of type 2 diabetes' is justified by the evidence presented.

First, let me verify the logical structure of the argument. The study establishes a correlation between coffee consumption and lower diabetes rates, then concludes causation. This is potentially a post hoc ergo propter hoc fallacy. Let me check if the evidence supports the stronger causal claim.

Examining the methodology, the study is observational rather than experimental, which limits causal inference. Let me verify whether they controlled for confounding variables. The paper mentions controlling for age, BMI, and physical activity, which are indeed major confounders. However, I should check if other important confounders were addressed: dietary patterns, socioeconomic status, and family history of diabetes. Reviewing the methods section again, I see they controlled for dietary patterns and education level (as a proxy for socioeconomic status) but not family history. This is a limitation in their confounder control.

Now let me verify the statistical analysis. The reported association shows a relative risk reduction of 27% (95% CI: 18-35%) for those consuming ≥3 cups daily versus non-drinkers. The p-value is <0.001, indicating statistical significance by conventional standards. Let me check if they addressed multiple hypothesis testing, as they analyzed several exposure categories... reviewing again, they did apply Bonferroni correction, which is appropriate.

Let me verify biological plausibility. The authors propose that coffee's antioxidant compounds improve insulin sensitivity. Checking against established literature, there is indeed preliminary evidence from laboratory studies supporting this mechanism, strengthening their causal argument.

Let me also verify consistency with other evidence. The authors cite three other large cohort studies with similar findings, which increases confidence through replication. However, I should note they don't mention any contradictory studies, which I need to verify against the broader literature... researching further, there are two smaller studies with null findings they didn't cite.

Based on this verification process, I conclude that while there is a robust association between coffee consumption and reduced diabetes risk that meets statistical standards and has some biological plausibility, the causal claim is stronger than warranted given the observational design and incomplete control of confounders. A more accurate conclusion would be that 'coffee consumption is associated with reduced risk of type 2 diabetes.'"

Annotation:
```json
{{
  "explanation": "The participant demonstrates comprehensive verification throughout its reasoning process by systematically checking the logical structure of the argument, evaluating the methodology against scientific standards, verifying statistical analysis procedures, assessing biological plausibility against established knowledge, confirming consistency with broader literature, and identifying limitations in the evidence that affect the validity of the conclusion.",
  "span_analysis": "Analyzing this text for verification: The phrase 'First, let me verify the logical structure of the argument' explicitly shows verification through validity checking. The text 'Let me check if the evidence supports the stronger causal claim' demonstrates confirmation procedures. The phrase 'Let me verify whether they controlled for confounding variables' shows methodological verification. The text 'Now let me verify the statistical analysis' demonstrates procedural verification. The phrase 'Let me verify biological plausibility' shows evidence verification. The text 'Let me also verify consistency with other evidence' shows confirmation against external standards. These show comprehensive verification across multiple dimensions.",
  "spans": [[143, 202], [375, 439], [552, 616], [1077, 1120], [1529, 1567], [1820, 1871]],
  "score": 2
}}
```

Now it's your turn, you are provided with a question and the participant's response with their reasoning process. You should carefully analyze the reasoning trace and provide your professional and scientific judgment of whether the cognitive capability of verification is present in the reasoning, regardless of the correctness of the response. Provide your analysis in the strict JSON format specified above.

Question: {question}

Reasoning trace: {response}

Annotation:"""

# =============================================================================
# GUIDANCE TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Algorithmic_7.txt
# -----------------------------------------------------------------------------
ALGORITHMIC_7_GUIDANCE_TEMPLATE = r"""You are solving an algorithmic problem, which involves applying a specific procedure, formula, or series of steps to reach a solution. Algorithmic problems have well-defined methods and predictable solution paths.

# Problem to Solve
{question}

# Reasoning Structure for Algorithmic Problems

Follow this structured reasoning approach optimized for algorithmic problems:

## Phase 1: Selective Attention
**Goal**: Identify the relevant information and procedure needed.
- Extract the key values, variables, and constraints from the problem
- Identify which formula, algorithm, or procedure applies
- Filter out extraneous information that doesn't affect the solution
- Focus on what's given and what needs to be found

↓ Then ensure logical consistency ↓

## Phase 2: Logical Coherence
**Goal**: Establish a logically sound approach before executing steps.
- Confirm the selected procedure is appropriate for this problem type
- Ensure your understanding of the problem is internally consistent
- Verify that your approach logically connects inputs to desired outputs
- Check that each planned step will follow logically from the previous one

↓ Then execute the procedure ↓

## Phase 3: Sequential Organization
**Goal**: Execute the algorithm/procedure in the correct step-by-step order.
- Follow the prescribed sequence of operations (order matters!)
- Ensure each step is completed before moving to the next
- Build solutions progressively, with each step depending on prior results
- Maintain clear progression from initial values through transformations to final answer

↓ Then validate your work ↓

## Phase 4: Verification
**Goal**: Check your procedural execution against established criteria.
- Verify each step was performed correctly according to the algorithm
- Confirm calculations or operations meet the procedural standards
- Check that the final answer satisfies the problem requirements
- Validate that you followed the procedure faithfully without skipping steps

↓ Then assess solution quality ↓

## Phase 5: Self-Evaluation (containing Pattern Recognition, which contains Decomposition and Integration)
**Goal**: Assess the quality and correctness of your solution process.

Evaluate whether your execution was accurate and efficient. Within this evaluation:

**Embedded Behavior - Pattern Recognition**: Identify recurring structures in the problem
- Recognize if this problem follows a familiar algorithmic pattern
- Identify common structural elements (e.g., iteration, recursion, formula application)
- Notice similarities to other problems you've solved
- Use pattern recognition to confirm you've applied the right procedure

**Within Pattern Recognition - Decomposition and Integration**: Break down and synthesize the solution
- Decompose the algorithm into its component steps or sub-procedures
- Solve each component independently if needed
- Integrate partial results to form the complete solution
- Ensure components work together coherently in the final answer

---

# Your Task
Apply this complete reasoning structure to solve the algorithmic problem. Make your reasoning process explicit by:
1. Showing selective attention to relevant information and the applicable procedure
2. Establishing logical coherence in your approach
3. Executing steps sequentially in the correct order
4. Verifying correctness against procedural standards
5. Self-evaluating through pattern recognition and decomposition/integration

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Case_Analysis_7.txt
# -----------------------------------------------------------------------------
CASE_ANALYSIS_7_GUIDANCE_TEMPLATE = r"""You are solving a **case analysis problem**, which involves interpreting a complex real-world 
situation with multiple interacting elements, ambiguous information, and overlapping goals. 
Case analysis requires identifying the most relevant details, structuring them, abstracting 
general principles, tracing causal and temporal relationships, and synthesizing a coherent, 
logically grounded interpretation or recommendation.

# Problem to Solve
{question}

# Reasoning Structure for Case Analysis Problems

Follow this optimized reasoning structure. The flow of reasoning is:

**selective-attention → sequential-organization → abstraction → causal-organization → 
temporal-organization → conceptual-level-processing → logical-coherence**

---

## Phase 1: Selective Attention
**Goal**: Identify the most relevant details from a complex, information-rich scenario.
- Extract the essential facts, stakeholders, constraints, and events
- Ignore tangential or extraneous details that obscure the central dynamics
- Identify patterns of relevance and key signals within the case
- Distinguish between primary drivers and secondary background information

↓ Organize what you’ve selected into a structured reasoning path ↓

## Phase 2: Sequential Organization
**Goal**: Build a clear analytical workflow for understanding the case.
- Break down the case into sequential steps: describe → analyze → interpret → evaluate
- Ensure each step logically builds on the previous one
- Maintain a structured progression from concrete details to higher-level insight
- Keep the analysis transparent and cumulative

↓ Once structured, move from specifics to general principles ↓

## Phase 3: Abstraction
**Goal**: Identify underlying patterns, themes, and governing principles.
- Generalize from case-specific details to conceptual categories
- Extract broader concepts (e.g., risks, incentives, constraints, systemic factors)
- Treat the case as an instance of a broader class of situations
- Use abstraction to clarify what is essential versus situational

↓ Use these abstracted elements to map out causal relationships ↓

## Phase 4: Causal Organization
**Goal**: Construct a causal explanation of how the case dynamics unfolded.
- Identify the causes, mechanisms, and effects linking the key elements
- Build a clear cause-and-effect chain connecting conditions to outcomes
- Determine which factors were driving forces and which were reactive
- Distinguish between direct causes, enabling conditions, and indirect influences

↓ Anchor the causal narrative in the actual order of events ↓

## Phase 5: Temporal Organization
**Goal**: Align your causal explanation with the sequence of events.
- Reconstruct the timeline of actions, decisions, and consequences
- Ensure causes precede effects and the temporal order is realistic
- Identify time-sensitive dependencies or delays that shaped the outcome
- Detect temporal patterns (e.g., escalation, feedback, lag effects)

↓ Translate the timeline into deeper conceptual insight ↓

## Phase 6: Conceptual-Level Processing
**Goal**: Interpret the case using high-level concepts or theories.
- Connect the case to broader conceptual frameworks (e.g., organizational theory, 
  behavioral principles, risk models, systemic dynamics)
- Explain the deeper meaning or significance of the events
- Use conceptual reasoning to evaluate decisions or outcomes
- Clarify the general principles the case illustrates

↓ Ensure the entire analysis forms a coherent whole ↓

## Phase 7: Logical Coherence
**Goal**: Ensure your final explanation is internally consistent and well-supported.
- Check that all claims align with the evidence in the case
- Verify that your abstractions, causal story, and conceptual insights do not contradict one another
- Ensure the final interpretation logically follows from the earlier steps
- Present a cohesive and rationally connected analysis

---

# Your Task
Apply this full reasoning structure to analyze the case. Make your reasoning explicit by:
1. Identifying relevant details through selective attention  
2. Structuring the analysis sequentially  
3. Abstracting broader patterns and principles  
4. Organizing the case causally  
5. Anchoring causes and effects temporally  
6. Interpreting the case using conceptual insight  
7. Checking for complete logical coherence  

Then present your final integrated case analysis.

# Output Format
Provide your response in the following JSON format:
{
  "final_answer": "your final answer here"
}"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Decision-Making_7.txt
# -----------------------------------------------------------------------------
DECISION_MAKING_7_GUIDANCE_TEMPLATE = r"""You are solving a decision-making problem, which involves selecting a single optimal option from a set of alternatives based on specific criteria. Decision-making problems require systematic evaluation and comparison to justify the best choice.

# Problem to Solve
{question}

# Reasoning Structure for Decision-Making Problems

Follow this structured reasoning approach optimized for decision-making problems:

## Phase 1: Selective Attention
**Goal**: Identify the decision criteria, alternatives, and relevant information.
- Extract the key criteria for evaluating options
- Identify all available alternatives to choose from
- Filter out information that doesn't affect the decision
- Focus on factors that differentiate between options

↓ Then establish logical foundations ↓

## Phase 2: Logical Coherence
**Goal**: Ensure your decision framework is internally consistent.
- Verify that evaluation criteria don't contradict each other
- Establish a logically sound approach to comparing alternatives
- Ensure your reasoning about trade-offs is consistent
- Check that your decision logic flows coherently from criteria to choice

↓ Then generate evaluation dimensions ↓

## Phase 3: Productivity
**Goal**: Generate comprehensive ways to evaluate and compare alternatives.
- Create multiple evaluation frameworks or perspectives for comparing options
- Generate diverse criteria combinations to assess alternatives
- Produce varied scenarios or weightings to test decision robustness
- Develop novel ways to think about trade-offs between options

↓ Then manage decision objectives ↓

## Phase 4: Goal Management
**Goal**: Track and balance multiple decision objectives.
- Maintain focus on the primary decision goal throughout evaluation
- Manage competing objectives and trade-offs between criteria
- Track which criteria are most important for the decision
- Adjust priorities if new insights reveal different goal emphases

|| While simultaneously structuring the process ||

## Phase 5: Sequential Organization (parallel)
**Goal**: Structure your decision process in logical evaluation stages.
- Organize your reasoning into phases (e.g., criteria identification → option evaluation → comparison → selection)
- Ensure systematic progression through evaluation steps
- Build your decision incrementally through ordered analysis
- Establish clear progression from alternatives to final choice

↓ Both converge to domain-aligned reasoning ↓

## Phase 6: Knowledge-Structure Alignment
**Goal**: Align your decision-making with domain-specific frameworks and conventions.
- Apply decision-making frameworks appropriate to this domain (e.g., cost-benefit for business, risk-benefit for medical)
- Use domain-standard criteria and evaluation methods
- Structure your reasoning according to how experts in this field make similar decisions
- Leverage domain-specific knowledge about what matters in these choices

|| While simultaneously thinking conceptually ||

## Phase 7: Conceptual-Level Processing (parallel)
**Goal**: Reason with abstract decision concepts before verbalizing specifics.
- Build mental models of how each alternative would play out
- Reason abstractly about trade-off structures before detailing specifics
- Understand options at a conceptual level (what they fundamentally represent)
- Think through implications conceptually before committing to detailed analysis

---

# Your Task
Apply this complete reasoning structure to solve the decision-making problem. Make your reasoning process explicit by:
1. Showing selective attention to decision criteria and alternatives
2. Establishing logical coherence in your evaluation framework
3. Generating productive evaluation approaches
4. Managing decision goals while organizing evaluation sequentially
5. Aligning with domain decision-making conventions while processing conceptually

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Design_7.txt
# -----------------------------------------------------------------------------
DESIGN_7_GUIDANCE_TEMPLATE = r"""You are solving a design problem, which involves creating solutions for vague goal statements with few constraints. Design problems require you to structure the problem space itself while generating innovative solutions.

# Problem to Solve
Critically evaluate the neutron detection technique described in a research paper on Low Energy Nuclear Reactions (LENR), considering the design of the detector, the statistical analysis of the data, and the potential sources of background noise. How would you improve the experiment to increase the confidence in the results, and what control measurements would you suggest to validate the findings?

# Reasoning Structure for Design Problems

Follow this structured reasoning approach optimized for design problems:

## Phase 1: Selective Attention
**Goal**: Identify what matters most in this vague problem space.
- Filter the essential requirements from the noise
- Distinguish core constraints from flexible elements  
- Identify key stakeholders, contexts, or success criteria
- Focus on what's truly given vs. what you need to define

↓ Then move to structured exploration ↓

## Phase 2: Sequential Organization (containing Productivity)
**Goal**: Build your solution through ordered stages of generative thinking.

Organize your design process into clear sequential stages (e.g., understanding → ideation → refinement → specification). Within each stage:

**Embedded Behavior - Productivity**: Generate diverse design alternatives
- Create multiple novel combinations of design elements
- Explore different approaches by recombining constraints, features, and concepts
- Produce varied solutions from the limited set of requirements and resources
- Don't settle on the first idea—systematically generate alternatives

↓ Then validate your reasoning ↓

## Phase 3: Logical Coherence
**Goal**: Ensure your design reasoning is internally consistent.
- Check that design decisions logically support each other
- Verify your solution doesn't contain contradictions
- Ensure constraints are satisfied without conflicts
- Validate that your reasoning chain from problem to solution is sound

|| While simultaneously ||

## Phase 4: Context-Awareness (containing Adaptive Detail Management, which contains Representational Restructuring)
**Goal**: Align your design thinking with the specific problem context and adjust as needed.

Match your reasoning to the design domain's characteristics and norms. Within this context-sensitive reasoning:

**Embedded Behavior - Adaptive Detail Management**: Adjust detail levels throughout your design process
- Start high-level (overall concept), then zoom into specifics where needed
- Keep some aspects abstract while detailing critical components
- Expand on complex elements, compress straightforward ones
- Shift between overview and details as the design develops

**Within Detail Management - Representational Restructuring**: Reformulate your design representation when stuck
- If one framing doesn't work, reframe the problem from a different angle
- Transform representations to reveal hidden insights (e.g., spatial → functional → temporal views)
- Restructure when you hit dead ends to make the problem more tractable
- Use alternative perspectives to unlock creative solutions

---

# Your Task
Apply this complete reasoning structure to solve the design problem. Make your reasoning process explicit by:
1. Showing your selective attention to essential elements
2. Organizing your design exploration sequentially while generating multiple alternatives
3. Checking logical coherence of your design decisions
4. Demonstrating context-awareness while adapting detail levels and restructuring representations when needed

Produce a well-reasoned design solution that reflects this structured approach.

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Diagnosis-Solution_7.txt
# -----------------------------------------------------------------------------
DIAGNOSIS_SOLUTION_7_GUIDANCE_TEMPLATE = r"""You are solving a **diagnosis–solution problem**, which involves (1) diagnosing the underlying cause 
of an issue and (2) proposing a justified solution tailored to the diagnosed cause. These problems 
require identifying core signals, aligning them with relevant knowledge structures, reasoning 
forward from possible causes, and selecting a strategy that leads to a coherent, conceptually grounded 
solution.

# Problem to Solve
{question}

# Reasoning Structure for Diagnosis–Solution Problems

Follow this optimized reasoning structure. The flow of reasoning is:

**selective-attention → sequential-organization → knowledge-structure-alignment → forward-chaining 
(containing strategy-selection) → strategy-selection → conceptual-level-processing ∥ causal-organization**

---

## Phase 1: Selective Attention
**Goal**: Identify the meaningful diagnostic cues in the problem.
- Extract key symptoms, constraints, or signals
- Separate high-value information from noise
- Highlight the central issue requiring explanation or correction
- Identify what is missing or ambiguous in the problem description

↓ Use the extracted information to structure your path forward ↓

## Phase 2: Sequential Organization
**Goal**: Create an ordered diagnostic–solution workflow.
- Break the problem into steps: observe → diagnose → infer → propose solution
- Ensure each step logically depends on the previous one
- Move cleanly from evidence to diagnosis, and from diagnosis to solution
- Maintain a transparent, stepwise flow

↓ Connect the sequence to relevant knowledge structures ↓

## Phase 3: Knowledge Structure Alignment
**Goal**: Match identified signs and steps to relevant conceptual frameworks.
- Map symptoms or observations onto known categories, models, or principles
- Align the problem with diagnostic schemas (e.g., known failure modes, conceptual taxonomies)
- Identify where the problem fits within established knowledge domains
- Use structured knowledge to constrain possible diagnoses

↓ Use aligned knowledge to reason forward through consequences ↓

## Phase 4: Forward Chaining (containing Strategy Selection)
**Goal**: Push implications forward from hypothesized causes to check consistency.
- Infer what should follow if a hypothesized cause is correct
- Compare those predicted outcomes to the observed facts
- Narrow down plausible diagnoses based on consistency
- Strengthen or eliminate hypotheses through logical propagation

Within this forward reasoning:

### Embedded Behavior — Strategy Selection
**Goal**: Choose the most suitable diagnostic and solution strategies.
- Identify which reasoning or intervention approach best matches the aligned knowledge structure
- Consider different solution pathways and evaluate their feasibility
- Select the strategy that offers the highest diagnostic clarity or problem resolution
- Justify why this approach is preferable to alternatives

↓ Develop a conceptually deep and coherent explanation ↓

## Phase 5: Conceptual-Level Processing (parallel with Causal Organization)
**Goal**: Ground your diagnosis and solution in core principles, not superficial symptoms.
- Explain the issue using underlying concepts or mechanisms
- Ensure the proposed solution addresses the conceptual root of the problem
- Generalize the reasoning when useful (e.g., principle-based explanation)
- Tie the diagnosis to higher-order understanding

|| In parallel ||

## Phase 6: Causal Organization
**Goal**: Construct a clear causal chain linking observations → causes → consequences → solution.
- Organize the explanation around cause-and-effect relationships
- Identify the initiating cause and trace its downstream effects
- Ensure your solution directly targets the identified causal pathway
- Present a cohesive causal narrative

---

# Your Task
Apply this full reasoning structure to solve the diagnosis–solution problem. Make your reasoning explicit by:
1. Identifying key signals using selective attention  
2. Structuring your workflow sequentially  
3. Aligning the problem with relevant knowledge structures  
4. Using forward chaining to test diagnostic hypotheses  
5. Selecting an appropriate strategy for solving the issue  
6. Demonstrating conceptual understanding of the diagnosis  
7. Organizing the explanation causally in parallel  

Then present your final, well-justified diagnosis and solution.

# Output Format
Provide your response in the following JSON format:
{
  "final_answer": "your final answer here"
}"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Dilemma_7.txt
# -----------------------------------------------------------------------------
DILEMMA_7_GUIDANCE_TEMPLATE = r"""You are solving a **dilemma problem**, which involves navigating a situation with two or more 
conflicting positions, each with meaningful trade-offs and no fully satisfactory resolution. 
Dilemmas require identifying the core tensions, generating alternatives, evaluating competing 
strategies, and reasoning through the causal implications of each pathway before forming an 
abstracted understanding of the underlying conflict.

# Problem to Solve
{question}

# Reasoning Structure for Dilemma Problems

Follow this optimized reasoning structure. The flow of reasoning is:

**selective-attention → sequential-organization → productivity → strategy-selection → 
forward-chaining → causal-organization → abstraction**

---

## Phase 1: Selective Attention
**Goal**: Identify the competing positions and essential tensions within the dilemma.
- Extract the core conflicting values, goals, or constraints
- Isolate the central trade-off(s) that make the problem a dilemma
- Remove details that do not materially influence the conflict
- Identify what must be balanced, sacrificed, or prioritized

↓ Organize these tensions into a structured reasoning sequence ↓

## Phase 2: Sequential Organization
**Goal**: Lay out a clear process for comparing and evaluating the conflicting options.
- Structure the analysis into steps: clarify positions → generate options → evaluate → integrate
- Ensure each step builds logically from the previous one
- Maintain a transparent progression through the dilemma
- Move from understanding to exploration to evaluation

↓ Use this structure to generate multiple meaningful options ↓

## Phase 3: Productivity
**Goal**: Generate a diverse set of ways to approach the dilemma.
- Produce multiple interpretations, options, or reframings of the problem
- Consider alternative solutions, compromises, or hybrid approaches
- Avoid prematurely narrowing down the options
- Expand the option space before evaluating strategies

↓ Use these options to select the most appropriate strategy ↓

## Phase 4: Strategy Selection
**Goal**: Choose the reasoning strategy best suited for this specific dilemma.
- Consider whether to compare consequences, principles, risks, or value systems
- Evaluate which strategy exposes the key differences between options
- Justify why this strategy is appropriate for resolving (or illuminating) this dilemma
- Select the strategy before reasoning through consequences

↓ Apply the chosen strategy to infer downstream implications ↓

## Phase 5: Forward Chaining
**Goal**: Trace the implications of each position or option forward.
- Follow each option to its logical consequences
- Identify how each choice influences stakeholders, goals, risks, or constraints
- Evaluate whether any option reduces or worsens the dilemma
- Compare the implications across options to highlight trade-offs

↓ Organize these consequences into a coherent causal story ↓

## Phase 6: Causal Organization
**Goal**: Build a cause-and-effect map of the dilemma’s dynamics.
- Identify what causes which outcomes under each option
- Connect initial choices to intermediate effects and final consequences
- Clarify how competing values or constraints generate the conflict
- Distinguish root causes from surface-level tensions

↓ Use the causal model to extract general insights ↓

## Phase 7: Abstraction
**Goal**: Generalize the dilemma to a higher-level understanding.
- Identify the deeper principle or structural conflict underlying the dilemma
- Abstract away from the specific scenario to a broader conceptual pattern
- Clarify what this dilemma reveals about competing values, systems, or priorities
- Highlight how the abstracted insight helps interpret or evaluate the dilemma

---

# Your Task
Apply this full reasoning structure to analyze the dilemma. Make your reasoning explicit by:
1. Identifying the central tensions through selective attention  
2. Structuring the analysis sequentially  
3. Generating multiple options via productivity  
4. Choosing an appropriate evaluative strategy  
5. Reasoning forward through the consequences of each option  
6. Organizing those consequences causally  
7. Abstracting the dilemma to reveal deeper insights  

Then present your final, integrated analysis of the dilemma.

# Output Format
Provide your response in the following JSON format:
{
  "final_answer": "your final answer here"
}"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Factual_Recall_7.txt
# -----------------------------------------------------------------------------
FACTUAL_RECALL_7_GUIDANCE_TEMPLATE = r"""You are solving a **factual recall problem**, which requires retrieving precise information, 
definitions, descriptions, or procedural facts from memory or a known knowledge source. 
Unlike interpretive or generative tasks, factual recall emphasizes accuracy, clarity, and 
alignment with established information structures.

# Problem to Solve
{question}

# Reasoning Structure for Factual Recall Problems

Follow this optimized reasoning structure. The flow of reasoning is:

**selective-attention → logical-coherence → forward-chaining → sequential-organization → 
knowledge-structure-alignment → adaptive-detail-management → temporal-organization**

---

## Phase 1: Selective Attention
**Goal**: Identify exactly what factual information the question is asking for.
- Determine the specific term, concept, definition, event, or procedural step being queried
- Ignore unrelated associations or interpretive tangents
- Focus on the minimal facts needed to answer accurately
- Distinguish primary information from supporting details

↓ Verify the factual content is consistent and error-free ↓

## Phase 2: Logical Coherence
**Goal**: Ensure your recalled information forms a consistent, non-contradictory statement.
- Check that the retrieved facts align with established knowledge
- Confirm that your explanation or definition is internally consistent
- Avoid mixing unrelated facts or concepts
- Keep the factual recall logically structured and precise

↓ Use this coherent structure to follow the logical chain of what the fact implies ↓

## Phase 3: Forward Chaining
**Goal**: Extend the factual recall through direct implications, subcomponents, or clarifications.
- Identify what the recalled fact logically entails
- Include any necessary supporting elements (e.g., properties, components, immediate consequences)
- Expand only as needed to provide a complete and accurate factual answer
- Ensure all additions follow directly from the recalled information

↓ Organize those elements into a clean factual sequence ↓

## Phase 4: Sequential Organization
**Goal**: Present the factual information in a clear, orderly manner.
- Break multi-part facts into logical steps, components, or layers
- Arrange information in a natural order (e.g., definition → features → example)
- Maintain clarity by structuring the recall from simple to specific
- Avoid presenting facts in a disordered or confusing way

↓ Align this sequence with established knowledge structures ↓

## Phase 5: Knowledge Structure Alignment
**Goal**: Fit the recalled information into its proper conceptual or categorical framework.
- Identify the domain the fact belongs to (e.g., biology, history, physics, math)
- Place the fact within its hierarchy: category → subcategory → instance
- Clarify how the fact relates to neighboring concepts
- Ensure the final explanation matches conventional organization in the field

↓ Adjust level of detail to match the question’s demands ↓

## Phase 6: Adaptive Detail Management
**Goal**: Provide the right amount of detail—neither too much nor too little.
- Expand details only when necessary for clarity or completeness
- Compress or omit extraneous specifics
- Adjust granularity based on whether the question asks for a definition, summary, list, or explanation
- Maintain a focus on the core factual content

↓ Place any temporally relevant facts in their proper time order ↓

## Phase 7: Temporal Organization
**Goal**: If the fact involves events or procedures, align them chronologically.
- Present historical or procedural facts in temporal order
- Ensure steps follow the correct sequence
- Maintain clarity if time is a defining part of the fact (e.g., “first… then…”)
- Use time-ordered reasoning only when applicable

---

# Your Task
Apply this full reasoning structure to recall the required factual information. Make your reasoning explicit by:
1. Identifying the exact fact being asked for  
2. Ensuring the recalled content is logically coherent  
3. Extending the fact via forward chaining only as needed  
4. Structuring the information sequentially  
5. Aligning the content with the relevant knowledge framework  
6. Adjusting the level of detail appropriately  
7. Incorporating temporal order when applicable  

Then provide the final factual answer.

# Output Format
Provide your response in the following JSON format:
{
  "final_answer": "your final answer here"
}"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Logical_7.txt
# -----------------------------------------------------------------------------
LOGICAL_7_GUIDANCE_TEMPLATE = r"""You are solving a **logical problem**, which involves reasoning about abstract relationships,
constraints, and implications in a precise and internally consistent way. Logical problems do not 
rely on domain knowledge; instead, they require you to detect structure, apply rules, and derive 
conclusions from the given information alone.

# Problem to Solve
{question}

# Reasoning Structure for Logical Problems

Follow this structured reasoning approach optimized for logical problems. Each behavior contributes
to rigorous, step-by-step inference. The flow of reasoning is:

**selective-attention → logical-coherence → pattern-recognition → sequential-organization**
(while sequential-organization runs in parallel with adaptive-detail-management → compositionality → productivity)

---

## Phase 1: Selective Attention
**Goal**: Identify only the information that matters for the logical inference.
- Filter out irrelevant or distracting details
- Identify the propositions, conditions, and constraints that define the logical structure
- Clarify what is explicitly given vs. what must be inferred

↓ Proceed to constructing a consistent reasoning foundation ↓

## Phase 2: Logical Coherence
**Goal**: Build an internally consistent interpretation of the given information.
- Align all premises so they do not contradict one another
- Ensure each inference follows validly from prior steps
- Maintain strict adherence to logical rules, not assumptions

↓ Use coherence to surface deeper structures ↓

## Phase 3: Pattern Recognition
**Goal**: Detect structural patterns in the information.
- Notice repeated forms, symmetries, equivalences, or contrasts
- Identify conditional structures (if–then), logical operators, or relational patterns
- Use these patterns to identify the types of inferences that are possible

↓ Transform patterns into an ordered reasoning path ↓

## Phase 4: Sequential Organization (parallel with Adaptive Detail Management)
**Goal**: Lay out your inference steps in a clear, logically ordered progression.
- Break the reasoning process into discrete steps from premises → intermediate conclusions → final conclusion
- Ensure each step depends only on already-established information
- Present deductions in a clean, cumulative sequence

|| In parallel ||

## Phase 5: Adaptive Detail Management (containing Compositionality → Productivity)
**Goal**: Adjust the level of detail as needed during reasoning.
- Zoom in on complex sub-steps when necessary
- Zoom out when details are trivial or redundant
- Allocate detail to components that affect the overall logical chain

Within this process:

### Embedded Behavior — Compositionality
**Goal**: Build complex logical structures out of simple components.
- Combine atomic facts into compound propositions
- Decompose complex statements into simpler parts
- Integrate small inferences into larger, multi-step reasoning chains

And within compositionality:

### Embedded Behavior — Productivity
**Goal**: Generate all logically valid possibilities or pathways.
- Enumerate all candidate interpretations consistent with the premises
- Explore alternate chains of inference when applicable
- Do not stop at the first conclusion—ensure completeness by considering all meaningful permutations

---

# Your Task
Apply this complete reasoning structure to solve the logical problem. Make your reasoning process explicit by:
1. Demonstrating selective attention to the key logical elements
2. Maintaining logical coherence throughout the inference
3. Identifying structural patterns that guide the reasoning
4. Laying out your deductions sequentially
5. Managing detail levels appropriately while composing complex structures from simpler pieces
6. Generating all relevant logical possibilities before committing to a conclusion

Then provide the final logically derived answer.

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Rule-Using_7.txt
# -----------------------------------------------------------------------------
RULE_USING_7_GUIDANCE_TEMPLATE = r"""You are solving a rule-using problem, which has a clear purpose or goal that is constrained by rules or principles but not restricted to a single specific procedure. Rule-using problems require applying domain rules flexibly to achieve well-defined objectives.

# Problem to Solve
{question}

# Reasoning Structure for Rule-Using Problems

Follow this structured reasoning approach optimized for rule-using problems:

## Phase 1: Selective Attention
**Goal**: Identify the relevant rules, constraints, and goal specifications.
- Extract the key rules, principles, or constraints that apply
- Identify the target goal or objective to achieve
- Filter out information that doesn't affect which rules to apply
- Focus on the conditions that trigger or constrain rule application

↓ Then organize your approach ↓

## Phase 2: Sequential Organization
**Goal**: Structure your rule application process in logical stages.
- Plan the sequence of rule applications needed
- Organize your reasoning into ordered phases or steps
- Recognize that some rules must be applied before others
- Establish a progression from initial state toward the goal

|| While simultaneously managing multiple strategic aspects ||

## Phase 3: Backward Chaining (parallel)
**Goal**: Work backward from the goal to identify what rules must be satisfied.
- Start with the desired outcome and identify prerequisites
- Determine which conditions must be true for the goal to be achieved
- Recursively identify what rules need to be satisfied to meet each condition
- Work backward through dependencies to find a path from current state to goal

|| At the same time ||

## Phase 4: Goal Management (parallel)
**Goal**: Track and maintain focus on objectives throughout reasoning.
- Keep the primary goal active while working through rule applications
- Manage subgoals that emerge from rule constraints
- Monitor progress toward the objective as rules are applied
- Adjust goals if new information reveals better approaches

|| At the same time ||

## Phase 5: Spatial Organization (parallel)
**Goal**: Represent relationships between elements spatially where relevant.
- Organize problem elements according to spatial relationships if applicable (positions, arrangements, configurations)
- Track relative positions or layouts that affect rule application
- Consider spatial constraints imposed by rules
- Visualize spatial configurations that result from applying rules

↓ All parallel reasoning converges to ↓

## Phase 6: Context-Awareness (containing Decomposition and Integration)
**Goal**: Align your rule application with the specific problem domain and context.

Recognize domain-specific conventions for how rules are typically applied in this context. Within this context-sensitive reasoning:

**Embedded Behavior - Decomposition and Integration**: Break the problem into rule-governed subproblems
- Decompose the problem into components where different rules apply
- Solve each component by applying relevant rules independently
- Integrate partial solutions while ensuring rules are satisfied across boundaries
- Synthesize component results into a coherent solution that meets all constraints

---

# Your Task
Apply this complete reasoning structure to solve the rule-using problem. Make your reasoning process explicit by:
1. Showing selective attention to relevant rules and goals
2. Organizing your rule application sequentially
3. Simultaneously: working backward from goals, managing objectives, and considering spatial relationships
4. Demonstrating context-awareness while decomposing and integrating rule-governed components

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Story_Problem_7.txt
# -----------------------------------------------------------------------------
STORY_PROBLEM_7_GUIDANCE_TEMPLATE = r"""You are solving a story problem, which embeds a mathematical formula or procedure within a narrative context. Story problems require extracting quantitative information from descriptive text and applying appropriate procedures.

# Problem to Solve
{question}

# Reasoning Structure for Story Problems

Follow this structured reasoning approach optimized for story problems:

## Phase 1: Logical Coherence
**Goal**: Establish a logically consistent understanding of the problem scenario.
- Ensure your interpretation of the story context is internally consistent
- Verify that relationships between story elements make logical sense
- Confirm that the problem setup doesn't contain contradictions
- Establish a coherent mental model of the situation before proceeding

↓ Then organize your approach ↓

## Phase 2: Sequential Organization
**Goal**: Structure your solution process in a logical step-by-step order.
- Identify the sequence of operations needed (what must be calculated first, second, etc.)
- Recognize dependencies between steps (later calculations depend on earlier ones)
- Plan the order of your solution before executing it
- Ensure each step builds properly on previous results

↓ Then break down the problem ↓

## Phase 3: Decomposition and Integration
**Goal**: Break the story problem into manageable subproblems, solve each, then combine results.
- Identify distinct components or questions within the narrative
- Separate the problem into solvable parts (e.g., find rate, then time, then distance)
- Solve each subproblem independently
- Integrate partial solutions into a coherent final answer
- Ensure combined results answer the original question

↓ Then build understanding compositionally ↓

## Phase 4: Compositionality
**Goal**: Build complex understanding from simpler components.
- Understand how simple facts from the story combine into complex relationships
- Recognize how basic quantities compose into derived quantities (e.g., rate × time = distance)
- Build your solution by systematically combining simpler elements
- Understand the whole problem through its parts and how they relate

↓ Then focus on what matters ↓

## Phase 5: Selective Attention
**Goal**: Filter the narrative to focus on quantitatively relevant information.
- Identify the numbers, units, and relationships that matter for calculation
- Distinguish story details that affect the solution from decorative narrative elements
- Focus on the mathematical core hidden in the story context
- Ignore irrelevant descriptive details while retaining essential quantities

↓ Then derive the solution forward ↓

## Phase 6: Forward Chaining (containing Pattern Recognition)
**Goal**: Start from given information and progressively derive the answer.

Begin with the known quantities from the story and apply operations to derive new information step-by-step until reaching the goal. Within this forward progression:

**Embedded Behavior - Pattern Recognition**: Identify the underlying problem type and structure
- Recognize what kind of story problem this is (rate problem, work problem, mixture problem, etc.)
- Identify familiar patterns that indicate which formulas or procedures to use
- Notice structural similarities to other story problems you've encountered
- Use the recognized pattern to guide which operations to apply in your forward derivation

---

# Your Task
Apply this complete reasoning structure to solve the story problem. Make your reasoning process explicit by:
1. Establishing logical coherence in understanding the scenario
2. Organizing your solution steps sequentially
3. Decomposing into subproblems and integrating results
4. Building complex understanding from simple components
5. Selecting relevant quantitative information from the narrative
6. Forward chaining from givens while recognizing the problem pattern

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/guidance_templates/Troubleshooting_7.txt
# -----------------------------------------------------------------------------
TROUBLESHOOTING_7_GUIDANCE_TEMPLATE = r"""You are solving a **troubleshooting problem**, which involves diagnosing a fault state by 
systematically identifying symptoms, tracing causal chains, and isolating the most probable source 
of failure. Troubleshooting problems require a structured, evidence-driven process for explaining 
what went wrong and why.

# Problem to Solve
{question}

# Reasoning Structure for Troubleshooting Problems

Follow this structured reasoning approach optimized for troubleshooting. Each behavior contributes to 
building an accurate causal explanation. The flow of reasoning is:

**selective-attention → logical-coherence → pattern-recognition → sequential-organization → 
forward-chaining → temporal-organization → compositionality**

---

## Phase 1: Selective Attention
**Goal**: Identify the critical information relevant to diagnosing the fault.
- Extract the key symptoms, system behaviors, or anomalies
- Ignore irrelevant background details
- Identify which components, processes, or states are implicated
- Distinguish between primary symptoms and secondary noise

↓ Establish a logically consistent view of the situation ↓

## Phase 2: Logical Coherence
**Goal**: Build a consistent interpretation of all observed symptoms.
- Ensure that no hypothesis contradicts the known properties of the system
- Align all clues under a single coherent diagnostic frame
- Reject explanations that cannot fit all observed facts
- Maintain a strict cause-and-effect standard

↓ Use coherence to reveal diagnostic patterns ↓

## Phase 3: Pattern Recognition
**Goal**: Detect diagnostic patterns that suggest likely sources of failure.
- Identify recurring symptom clusters or known failure signatures
- Match patterns to prior faults, system models, or causal archetypes
- Notice inconsistencies or missing signals that may reveal system-level issues
- Use anomaly alignment to narrow the fault space

↓ Convert recognized patterns into an ordered reasoning pathway ↓

## Phase 4: Sequential Organization
**Goal**: Lay out a step-by-step diagnostic plan.
- Break the problem into ordered steps, from initial symptoms to deeper checks
- Reason from observable evidence → intermediate inferences → candidate fault hypotheses
- Ensure each step builds on validated information
- Maintain clarity in the progression of diagnostic reasoning

↓ Use the ordered plan to infer consequences through the system ↓

## Phase 5: Forward Chaining
**Goal**: Push implications forward through the system to test hypotheses.
- Given a hypothesis, infer what should happen next if it were true
- Compare expected consequences with actual system behavior
- Use mismatches to rule out incorrect hypotheses
- Use matches to strengthen the case for a likely fault source

↓ Anchor these forward inferences to the timeline of events ↓

## Phase 6: Temporal Organization
**Goal**: Align the fault explanation with the actual sequence of events.
- Construct a timeline linking events, system states, and symptoms
- Ensure that the fault explanation matches when each symptom occurred
- Identify causes that must precede effects and eliminate impossible sequences
- Use temporal consistency as a diagnostic filter

↓ Integrate the temporally coherent components into a full causal explanation ↓

## Phase 7: Compositionality
**Goal**: Build a complete fault explanation from smaller causal pieces.
- Combine local inferences into a system-level diagnosis
- Integrate component failures, temporal patterns, and causal implications
- Synthesize the most probable fault source that explains all symptoms
- Present a cohesive, well-structured causal model of the failure

---

# Your Task
Apply this complete reasoning structure to diagnose the troubleshooting problem. Make your reasoning 
process explicit by:
1. Identifying key symptoms using selective attention  
2. Building a logically coherent interpretation of all evidence  
3. Detecting relevant diagnostic patterns  
4. Organizing the diagnostic steps sequentially  
5. Using forward chaining to test hypotheses  
6. Structuring the timeline with temporal organization  
7. Composing all validated inferences into a final fault explanation  

Then present your final diagnosis clearly and concisely.

# Output Format
Provide your response in the following JSON format:
```json
{
  "final_answer": "your final answer here"
}
```"""

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Algorithmic_7.txt
# -----------------------------------------------------------------------------
ALGORITHMIC_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Algorithmic
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. selective-attention
2. logical-coherence
3. sequential-organization
4. compositionality
5. decomposition-and-integration
6. abstraction
7. adaptive-detail-management

Reasoning flow:
  - selective-attention → logical-coherence
  - logical-coherence → sequential-organization
  - sequential-organization → compositionality
  - compositionality (contains) decomposition-and-integration
  - decomposition-and-integration → abstraction
  - abstraction (parallel with) adaptive-detail-management

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Case Analysis_7.txt
# -----------------------------------------------------------------------------
CASE_ANALYSIS_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Case Analysis
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. goal-management
2. self-awareness
3. sequential-organization
4. selective-attention
5. forward-chaining
6. logical-coherence
7. causal-organization

Reasoning flow:
  - goal-management → self-awareness
  - self-awareness → sequential-organization
  - sequential-organization → selective-attention
  - selective-attention → forward-chaining
  - forward-chaining → logical-coherence
  - logical-coherence → causal-organization

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Decision-Making_7.txt
# -----------------------------------------------------------------------------
DECISION_MAKING_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Decision-Making
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. compositionality
2. sequential-organization
3. logical-coherence
4. conceptual-level-processing
5. context-alignment
6. self-evaluation
7. self-awareness

Reasoning flow:
  - compositionality → sequential-organization
  - sequential-organization → logical-coherence
  - logical-coherence → conceptual-level-processing
  - conceptual-level-processing → context-alignment
  - context-alignment → self-evaluation
  - self-evaluation (contains) self-awareness

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Design_7.txt
# -----------------------------------------------------------------------------
DESIGN_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Design
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. logical-coherence
2. selective-attention
3. sequential-organization
4. forward-chaining
5. compositionality
6. abstraction
7. representational-restructuring

Reasoning flow:
  - logical-coherence (contains) selective-attention
  - selective-attention → sequential-organization
  - sequential-organization → forward-chaining
  - forward-chaining → compositionality
  - compositionality → abstraction
  - abstraction → representational-restructuring

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Diagnosis-Solution_7.txt
# -----------------------------------------------------------------------------
DIAGNOSIS_SOLUTION_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Diagnosis-Solution
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. compositionality
2. logical-coherence
3. knowledge-structure-alignment
4. causal-organization
5. conceptual-level-processing
6. context-alignment
7. backward-chaining

Reasoning flow:
  - compositionality → logical-coherence
  - logical-coherence → knowledge-structure-alignment
  - knowledge-structure-alignment → causal-organization
  - causal-organization → conceptual-level-processing
  - conceptual-level-processing → context-alignment
  - context-alignment → backward-chaining

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Dilemma_7.txt
# -----------------------------------------------------------------------------
DILEMMA_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Dilemma
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. goal-management

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Rule-Using_7.txt
# -----------------------------------------------------------------------------
RULE_USING_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Rule-Using
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. selective-attention
2. knowledge-structure-alignment
3. logical-coherence
4. goal-management
5. self-evaluation
6. backtracking
7. compositionality

Reasoning flow:
  - selective-attention → knowledge-structure-alignment
  - knowledge-structure-alignment → logical-coherence
  - logical-coherence → goal-management
  - goal-management → self-evaluation
  - self-evaluation (parallel with) backtracking
  - backtracking → compositionality

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Story Problem_7.txt
# -----------------------------------------------------------------------------
STORY_PROBLEM_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Story Problem
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. verification
2. self-evaluation
3. goal-management
4. logical-coherence
5. compositionality
6. forward-chaining
7. sequential-organization

Reasoning flow:
  - verification → self-evaluation
  - self-evaluation → goal-management
  - goal-management → logical-coherence
  - logical-coherence → compositionality
  - compositionality → forward-chaining
  - forward-chaining → sequential-organization

Please output your prompt template as a Python string."""

# -----------------------------------------------------------------------------
# FILE: structure_guidance/prompt_templates/Troubleshooting_7.txt
# -----------------------------------------------------------------------------
TROUBLESHOOTING_7_PROMPT_TEMPLATE = r"""We would like to prompt a model once to answer a question by utilizing this optimal reasoning structure to "structure"/"scaffold" its reasoning.

The different problem types that we are using vary from well-defined on the top to ill-defined on the bottom (Jonassen, 2000, p. 74), as follows:
* Logical Problems: abstract tests of reasoning that puzzle the learner
* Algorithmic Problems: repeating a series of steps through a procedure or formula
* Story Problems: story with formula or procedure embedded
* Rule-Using Problems: clear purpose or goal that is constrained but not restricted to a specific procedure or method
* Decision-Making Problems: selecting a single option from a set of alternatives based on a set of criteria
* Troubleshooting Problems: fault state diagnosis
* Case-Analysis Problems: complex, leisure-time system with multiple ill-defined goals
* Design Problems: vague goal statement with few constraints; requires structuring
* Dilemmas: situation with contradictory positions
* New: Factual Recall: This type is not included in the original Jonassen paper but is used to describe problems which test your ability to remember specific information, theories, definitions, or procedural knowledge directly stated in a source.

Our behavior structure is a DAG, where each node is a reasoning behavior and the edges between nodes can either be: 'contains', 'parallel', or 'next'. This means that a reasoning behavior (e.g., hierarchical organization for structuring the reasoning process and associated knowledge) may itself have a different reasoning behavior contained within it (e.g., decomposition and integration). Parallel indicates the two behaviors occur moreorless simultaneously, and 'next' indicates that the behaviors occur one and another.

I want you to construct a single prompt template for a given problem type and its reasoning structure. The prompt template should:
* Briefly describe the reasoning structure easily to the model (including making each of the behaviors intuitive such that the model will understand how to execute the behavior within its reasoning for the specific problem type).
* Make the relationships between the behaviors clear to the model
* Contextualize the overall reasoning behavior structure to the given problem type
* It should be able to take in as input a provided question

Here is my problem type: Troubleshooting
Here is the optimal 7-node behavior structure:

Follow this reasoning structure when solving the problem:

Key reasoning elements to include:
1. selective-attention
2. logical-coherence
3. causal-organization
4. conceptual-level-processing
5. sequential-organization
6. knowledge-structure-alignment
7. hierarchical-organization

Reasoning flow:
  - selective-attention → logical-coherence
  - logical-coherence → causal-organization
  - causal-organization → conceptual-level-processing
  - conceptual-level-processing → sequential-organization
  - sequential-organization → knowledge-structure-alignment
  - knowledge-structure-alignment → hierarchical-organization

Please output your prompt template as a Python string."""

# =============================================================================
# DOCUMENTATION
# =============================================================================

# -----------------------------------------------------------------------------
# FILE: README.md
# -----------------------------------------------------------------------------
README_MD = r"""<div align="center">

<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/readme_image.png" alt="Cognitive Foundations"/></p>

[![Static Badge](https://img.shields.io/badge/Paper-white?style=for-the-badge&logo=arxiv&logoColor=%23e46e2f&color=%232e4969)](https://arxiv.org/abs/2511.16660)
[![Static Badge](https://img.shields.io/badge/Blog-white?style=for-the-badge&logo=notion&logoColor=%23e46e2f&color=%232e4969)](https://tinyurl.com/cognitive-foundations)
[![Static Badge](https://img.shields.io/badge/Dataset-white?style=for-the-badge&logo=huggingface&logoColor=%23e46e2f&color=%232e4969)](https://huggingface.co/collections/stellalisy/cognitive-foundations)

[Priyanka Kargupta*](https://pkargupta.github.io/), [Shuyue Stella Li*](https://stellalisy.com/), [Haocheng Wang](https://hassonlab.princeton.edu/publications/contributor/wang-haocheng), [Jinu Lee](https://jinulee-v.github.io/), [Shan Chen](https://shanchen.dev/), [Orevaoghene Ahia](https://orevaahia.github.io/), [Dean Light](https://www.linkedin.com/in/dean-light), [Thomas L. Griffiths](https://cocosci.princeton.edu/tom/index.php), [Max Kleiman-Weiner](https://faculty.washington.edu/maxkw/), [Jiawei Han](https://hanj.cs.illinois.edu/), [Asli Celikyilmaz](http://asli.us/), [Yulia Tsvetkov](https://homes.cs.washington.edu/~yuliats/)

_*Equal contribution in alphabetical order_

</div>

# Cognitive Foundations for Reasoning and Their Manifestation in LLMs

## Links

- [Overview](#overview)
  - [Installation](#installation)
- [Assessing Behavioral Manifestation of Cognitive Elements](#assessing-behavioral-manifestation-of-cognitive-elements)
  - [Output Data Format](#output-data-format)
- [Test-Time Reasoning Guidance](#test-time-reasoning-guidance)
  - [Generating Guidance Templates](#generating-guidance-templates)
- [Citations](#citation)


## Overview

Our framework bridges **cognitive science** and **large language model (LLM) research** to systematically understand how LLMs reason and to diagnose/improve their reasoning processes, based on analysis of 192K model traces and 54 human think-aloud traces.

### Installation
The code is written in Python 3.10.9. The Python dependencies are summarized in the file `requirements.txt`. You can install them like this:
```
pip install -r requirements.txt
```

## Assessing Behavioral Manifestation of Cognitive Elements

We develop a taxonomy of **28 cognitive elements** spanning reasoning goals & properties, meta-cognitive controls, reasoning & knowledge representations, and transformation operations, creating a shared vocabulary between cognitive science and LLM research. We utilize this framework to encode reasoning traces into a **heterogenous graph**, where each node represents a cognitive element and edges between them reflect their temporal and hierarchical relationships.

<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/taxonomy.png" alt="Cognitive Foundations"/></p>

Our evaluation encompasses **192K+ model traces** from **18 different LLMs** across text, vision, and audio modalities, alongside **54 human think-aloud traces** to enable direct comparison between human and machine reasoning patterns. We study both _well-structured_ (e.g., Algorithmic) to _ill-structured_ (e.g., Dilemma) problem types. We provide all span-level annotation prompts in `element_annotation`.

### Output Data Format

In order to run [test-time reasoning guidance](#test-time-reasoning-guidance), we expect the following JSON file format for each model's span-level annotation result. We automatically read all model-specific JSON files from a specified directory:

```
# One file per model
{
    "[question_id]_[model_name]": {
        "sample_id": "[question_id]_[model_name]",
        "question_id": [int: question_id],
        "task": [str: task],
        "model_name": [str: the name of the model],
        "problem_type": [either a string label of the problem type or a list of index ids (we will take the mode of the latter)],
        "correctness": [bool: whether the model's final answer is correct or incorrect],
        "element_annotation": {
            "[element_label]": {
                "score": [int: 0-2, where 0 indicates no element present, 1 for partially present, and 2 for strongly present],
                "spans": [list: each item is a list of length 2, indicating both the start and end span index]
            },
            ...
        }
    }
}
```

## Test-Time Reasoning Guidance

We introduce **test-time reasoning guidance** as a targeted intervention to explicitly scaffold cognitive patterns predictive of reasoning success. In greedy fashion, we determine the most success-prone reasoning structure (subgraph) for each problem type, based on our empirical analysis. We convert each into a prompt which guides a model's reasoning process, improving performance by up to 26.7% on ill-structured problems while maintaining baseline performance on well-structured ones.

### Generating Guidance Templates

To generate test-time reasoning guidance templates for different problem types, run the `construct_graphs.py` script:

```bash
python construct_graphs.py \
    --element_dir /path/to/span_annotations \
    --prompt_template_dir structure_guidance/prompt_templates \
    --output_dir reasoning_structure/output_consensus_graphs \
    --path_to_question_info /path/to/question_info.json \
    --max_nodes 7 \
    --overlap_threshold 0.8 \
    --parallel_threshold 20
```

**Arguments:**
- `--element_dir`: Directory containing span-level annotation files (in the format described above)
- `--prompt_template_dir`: Output directory for generated prompts (default: `structure_guidance/prompt_templates`)
- `--output_dir`: Output directory for consensus graph visualizations (default: `reasoning_structure/output_consensus_graphs`)
- `--path_to_question_info`: Path to JSON file containing question metadata
- `--max_nodes`: Maximum number of nodes in the consensus graph (default: 7)
- `--overlap_threshold`: Overlap threshold for span tree construction (default: 0.8)
- `--parallel_threshold`: Parallel threshold for span tree construction (default: 20)
- `--target_type`: Optional filter for specific problem type (default: processes all types)

**Output:**

The script generates prompts in `structure_guidance/prompt_templates` that can be input into any model (we used **Claude Sonnet 4.5**) to produce reasoning guidance templates. These templates are then used during test-time to scaffold the model's reasoning process.

**Example Resources:**
- Graph visualizations for each problem type (max_nodes=7): `reasoning_structure/output_consensus_graphs/7/`
- Generated prompts for constructing guidance templates: `structure_guidance/prompt_templates/`
- Final test-time guidance templates: `structure_guidance/guidance_templates/`

## Citation

```bibtex
@article{kargupta2025cognitive,
  title={Cognitive Foundations for Reasoning and Their Manifestation in LLMs},
  author={Kargupta, Priyanka and Li, Shuyue Stella and Wang, Haocheng and Lee, Jinu and Chen, Shan and Ahia, Orevaoghene and Light, Dean and Griffiths, Thomas L and Kleiman-Weiner, Max and Han, Jiawei and Celikyilmaz, Asli and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2511.16660},
  year={2025}
}
```"""

# -----------------------------------------------------------------------------
# FILE: requirements.txt
# -----------------------------------------------------------------------------
REQUIREMENTS_TXT = r"""json_repair==0.54.2
matplotlib==3.10.7
networkx==2.8.4
numpy==2.2.6
openai==2.8.1
pydantic==2.12.4
pygraphviz==1.14
scipy==1.13.0
tqdm==4.64.1
"""

# =============================================================================
# END OF COMBINED REPOSITORY FILE
# =============================================================================