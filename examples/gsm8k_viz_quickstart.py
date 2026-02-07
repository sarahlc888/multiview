#!/usr/bin/env python3
"""Quick start example for GSM8K computational graph visualization.

This script demonstrates the basic usage without requiring graphviz installation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiview.visualization.gsm8k_graph import (
    GSM8KComputationalGraph,
    parse_gsm8k_document,
)


def main():
    print("=" * 70)
    print("GSM8K Computational Graph - Quick Start Example")
    print("=" * 70)
    print()

    # Example GSM8K problem
    example_doc = """Question: Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?
Answer: The number of books borrowed on Friday is higher by 40 * 40/100 = <<40*40/100=16>>16 books.
There are 5 days from Monday to Friday inclusive, so Krystian borrows an average of 5 * 40 = <<5*40=200>>200 books during that time.
With Friday's increase in borrowings, during one week Krystian borrows 200 + 16 = <<200+16=216>>216 books.
#### 216"""

    # Parse the document
    print("1. Parsing GSM8K problem...")
    print("-" * 70)
    question, answer = parse_gsm8k_document(example_doc)

    print("QUESTION:")
    print(question)
    print()
    print("ANSWER:")
    print(answer)
    print()

    # Build computational graph
    print("2. Building computational graph...")
    print("-" * 70)
    graph = GSM8KComputationalGraph(question, answer)
    graph.parse()

    print("✓ Graph created successfully!")
    print(
        f"  - Input nodes: {sum(1 for n in graph.nodes.values() if n.node_type == 'input')}"
    )
    print(
        f"  - Calculation nodes: {sum(1 for n in graph.nodes.values() if n.node_type == 'operation')}"
    )
    print(
        f"  - Result nodes: {sum(1 for n in graph.nodes.values() if n.node_type == 'result')}"
    )
    print(f"  - Total edges: {len(graph.edges)}")
    print(f"  - Final answer: {graph.final_answer}")
    print()

    # Show graph structure
    print("3. Graph structure:")
    print("-" * 70)

    # Group nodes by type
    inputs = {k: v for k, v in graph.nodes.items() if v.node_type == "input"}
    calcs = {k: v for k, v in graph.nodes.items() if v.node_type == "operation"}
    result = {k: v for k, v in graph.nodes.items() if v.node_type == "result"}

    print("\nINPUT VALUES:")
    for node_id, node in sorted(inputs.items()):
        print(f"  [{node_id}] = {node.value}")

    print("\nCALCULATIONS:")
    for node_id, node in sorted(calcs.items()):
        # Find inputs to this calculation
        incoming = [e.from_node for e in graph.edges if e.to_node == node_id]
        incoming_str = ", ".join(incoming) if incoming else "none"
        print(f"  [{node_id}] {node.expression} = {node.value}")
        print(f"            (uses: {incoming_str})")

    print("\nFINAL RESULT:")
    for node_id, node in result.items():
        incoming = [e.from_node for e in graph.edges if e.to_node == node_id]
        incoming_str = ", ".join(incoming) if incoming else "none"
        print(f"  [{node_id}] = {node.value}")
        print(f"            (from: {incoming_str})")

    # Export to JSON
    print()
    print("4. Exporting graph structure...")
    print("-" * 70)
    graph_dict = graph.to_dict()

    output_file = Path(__file__).parent / "example_graph.json"
    with open(output_file, "w") as f:
        json.dump(graph_dict, f, indent=2)

    print(f"✓ Graph exported to: {output_file}")
    print()

    # Show how to visualize (if graphviz is installed)
    print("5. Next steps:")
    print("-" * 70)
    print("To create a visual graph (requires graphviz):")
    print()
    print("  1. Install visualization dependencies:")
    print("     pip install -e '.[viz]'")
    print()
    print("  2. Install graphviz system library:")
    print("     brew install graphviz  # macOS")
    print("     sudo apt-get install graphviz  # Linux")
    print()
    print("  3. Visualize problems:")
    print("     python scripts/analyze_corpus.py \\")
    print("       --mode gsm8k_graphs \\")
    print("       --input outputs/.../documents.jsonl \\")
    print("       --output viz/gsm8k \\")
    print("       --num 10")
    print()
    print("  4. Or use the Jupyter notebook:")
    print("     jupyter notebook notebooks/gsm8k_visualization.ipynb")
    print()
    print("See VIZ_GSM8K.md for full documentation.")
    print()


if __name__ == "__main__":
    main()
