"""Computational graph visualization for GSM8K problems.

This module parses GSM8K math word problems into computational graphs,
extracting the flow of calculations and visualizing them.
"""

import re
import webbrowser
from dataclasses import dataclass
from typing import Any

# Word to number mapping
WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    mpatches = None  # type: ignore
    FancyBboxPatch = None  # type: ignore
    FancyArrowPatch = None  # type: ignore


@dataclass
class ComputationNode:
    """A node in the computational graph."""

    node_id: str
    node_type: str  # 'input', 'operation', 'result'
    value: float | None = None
    expression: str | None = None
    description: str | None = None


@dataclass
class ComputationEdge:
    """An edge in the computational graph."""

    from_node: str
    to_node: str
    label: str | None = None


class GSM8KComputationalGraph:
    """Parses and visualizes GSM8K problems as computational graphs."""

    def __init__(self, question: str, answer: str):
        """Initialize with a GSM8K question and answer.

        Args:
            question: The word problem question
            answer: The step-by-step solution with calculations
        """
        self.question = question
        self.answer = answer
        self.nodes: dict[str, ComputationNode] = {}
        self.edges: list[ComputationEdge] = []
        self.final_answer: float | None = None

    def parse(self):
        """Parse the problem into a computational graph."""

        def _to_float(num_str: str) -> float:
            """Parse a numeric string that may contain commas."""
            return float(str(num_str).strip().replace(",", ""))

        def _normalize_expr(expr: str) -> str:
            """Normalize unicode math operators to ASCII equivalents."""
            return (
                expr.replace("×", "*")
                .replace("÷", "/")
                .replace("−", "-")
                .replace("–", "-")
                .replace("—", "-")
            )

        # Extract all calculations from the answer in the order they appear

        # Find all bracketed calculations with their positions
        calc_pattern = r"<<([^=]+)=([^>]+)>>"
        bracketed_calcs = []
        for match in re.finditer(calc_pattern, self.answer):
            expr, result = match.groups()
            bracketed_calcs.append(
                {
                    "pos": match.start(),
                    "expr": _normalize_expr(expr.strip()),
                    "result": result.strip(),
                    "type": "bracketed",
                }
            )

        # Find all plain text calculations with their positions
        # Look for patterns like: number op number = result
        plain_calc_pattern = (
            r"([-+]?\d[\d,]*(?:\.\d+)?)\s*([-+*/×÷−–—])\s*"
            r"([-+]?\d[\d,]*(?:\.\d+)?)\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?)"
        )

        plain_calcs = []
        for match in re.finditer(plain_calc_pattern, self.answer):
            num1, op, num2, result = match.groups()
            match_start, _match_end = match.span()

            # Skip if this match is inside or very close to a bracketed calculation
            too_close = False
            for bcalc in bracketed_calcs:
                # Find the end position of this bracketed calc
                bracket_match = re.search(r"<<[^>]+>>", self.answer[bcalc["pos"] :])
                if bracket_match:
                    bracket_end = bcalc["pos"] + bracket_match.end()
                    # Check if plain calc overlaps or is within 30 chars
                    if (
                        match_start >= bcalc["pos"] - 30
                        and match_start <= bracket_end + 30
                    ):
                        too_close = True
                        break

            if not too_close:
                expr = _normalize_expr(f"{num1}{op}{num2}")
                plain_calcs.append(
                    {
                        "pos": match_start,
                        "expr": expr,
                        "result": result,
                        "type": "plain",
                    }
                )

        # Combine and sort by position in text
        all_calcs = bracketed_calcs + plain_calcs
        all_calcs.sort(key=lambda x: x["pos"])

        # Remove duplicates (same expr and result)
        seen = set()
        calculations = []
        for calc in all_calcs:
            # Results sometimes include commas (e.g. "1,200")
            result_str = str(calc["result"]).strip().replace(",", "")
            key = (calc["expr"], _to_float(result_str))
            if key not in seen:
                calculations.append((calc["expr"], result_str))
                seen.add(key)

        # Extract final answer (marked with ####)
        # GSM8K sometimes uses commas and/or signed numbers (rare)
        final_pattern = r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"
        final_match = re.search(final_pattern, self.answer)
        if final_match:
            self.final_answer = _to_float(final_match.group(1))

        # Extract numeric values from the question as initial inputs
        input_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", self.question)

        # Also extract word numbers (e.g., "two", "twelve") from question
        question_lower = self.question.lower()
        for word, num in WORD_TO_NUM.items():
            # Look for whole word matches
            if re.search(r"\b" + word + r"\b", question_lower):
                input_numbers.append(str(num))

        # Extract numbers from answer that appear outside of calculations
        # These are often constants or inferred values (e.g., "5 days", percentages)
        # Remove all <<...>> bracketed calculations from answer
        answer_no_calcs = re.sub(r"<<[^>]+>>", "", self.answer)
        # Also remove the final answer line
        answer_no_calcs = re.sub(r"####.*$", "", answer_no_calcs)
        # Now extract numbers from what remains
        answer_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", answer_no_calcs)

        # Filter out calculation results from being treated as inputs
        calc_results = {_to_float(result) for _, result in calculations}
        if self.final_answer is not None:
            calc_results.add(self.final_answer)

        for num_str in answer_numbers:
            num_val = _to_float(num_str)
            # Only add if it's not a calculation result
            if num_val not in calc_results:
                input_numbers.append(num_str)

        # Build deterministic input nodes keyed by numeric value so that we
        # can match "1" vs "1.0" (and other formatting differences).
        input_nodes: dict[float, str] = {}
        unique_input_vals = sorted({_to_float(n) for n in input_numbers})
        for i, val in enumerate(unique_input_vals):
            node_id = f"input_{i}"
            input_nodes[val] = node_id
            display_val: int | float = int(val) if float(val).is_integer() else val
            self.nodes[node_id] = ComputationNode(
                node_id=node_id,
                node_type="input",
                value=val,
                description=f"Value: {display_val}",
            )

        # Process each calculation step
        for i, (expr, result) in enumerate(calculations):
            node_id = f"calc_{i}"
            result_val = _to_float(result)

            # Clean up expression
            expr = _normalize_expr(expr.strip())

            self.nodes[node_id] = ComputationNode(
                node_id=node_id,
                node_type="operation",
                value=result_val,
                expression=expr,
                description=f"{expr} = {result}",
            )

            # Extract operands from the expression
            expr_numbers = re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", expr)

            # Try to match operands to available nodes
            # Prioritize: previous calc results > inputs > nothing
            matched_sources = []

            for num_str in expr_numbers:
                num_val = _to_float(num_str)

                # First, check if this matches a previous calculation result
                # (prefer using intermediate results over raw inputs)
                matched_id = None
                for prev_node_id, prev_node in self.nodes.items():
                    if (
                        prev_node.value is not None
                        and abs(prev_node.value - num_val) < 0.01
                        and prev_node_id != node_id
                        and prev_node.node_type == "operation"
                    ):
                        matched_id = prev_node_id
                        break

                # If no calc result matches, check inputs
                if matched_id is None:
                    for in_val, in_id in input_nodes.items():
                        if abs(in_val - num_val) < 0.01:
                            matched_id = in_id
                            break

                if matched_id is not None:
                    matched_sources.append(matched_id)

            # Create edges for matched sources
            # Remove consecutive duplicates (for cases like 40*40, only create one edge)
            seen = set()
            for source_id in matched_sources:
                if source_id not in seen:
                    self.edges.append(
                        ComputationEdge(from_node=source_id, to_node=node_id)
                    )
                    seen.add(source_id)

    def is_valid(self) -> bool:
        """Check if the computational graph is valid.

        A graph is valid if the plotted computations can actually give us the final answer.
        This means:
        - The final answer matches one of the extracted calculations
        - There is a path from inputs to the final answer through the computations
        """

        def _to_float(num_str: str) -> float:
            return float(str(num_str).strip().replace(",", ""))

        if self.final_answer is None:
            return False

        # Check if final answer appears as a calculation result
        calcs = [(k, v) for k, v in self.nodes.items() if v.node_type == "operation"]
        if not calcs:
            return False

        # Find which calculation produces the final answer
        final_calc_id = None
        for calc_id, calc in calcs:
            if abs(calc.value - self.final_answer) < 0.01:
                final_calc_id = calc_id
                break

        if final_calc_id is None:
            return False

        # Check if there's a path from inputs to the final calculation
        # Do a BFS/DFS to see if we can reach final_calc_id from any input
        reachable = set()

        # Start with all input nodes
        input_nodes = [k for k, v in self.nodes.items() if v.node_type == "input"]
        queue = list(input_nodes)
        reachable.update(input_nodes)

        # BFS to find all reachable nodes
        while queue:
            current = queue.pop(0)
            for edge in self.edges:
                if edge.from_node == current and edge.to_node not in reachable:
                    reachable.add(edge.to_node)
                    queue.append(edge.to_node)

        # Check if final calculation is reachable
        if final_calc_id not in reachable:
            return False

        # Check if all values used in calculations are accounted for
        # Build set of available values (inputs + all calc results + constants from question)
        available_values = set()

        # Add input values
        for node in self.nodes.values():
            if node.node_type == "input":
                available_values.add(node.value)

        # Add ALL calculation results (since they're all computed, just potentially out of order)
        for node in self.nodes.values():
            if node.node_type == "operation":
                available_values.add(node.value)

        # Extract all numbers from the question as potential constants
        question_numbers = re.findall(r"\b([-+]?\d[\d,]*(?:\.\d+)?)\b", self.question)
        for num_str in question_numbers:
            available_values.add(_to_float(num_str))

        # Extract all numbers from the answer as potential constants
        answer_numbers = re.findall(r"\b([-+]?\d[\d,]*(?:\.\d+)?)\b", self.answer)
        for num_str in answer_numbers:
            available_values.add(_to_float(num_str))

        # Now check if any calculation uses a value that's not available anywhere
        for i in range(len(calcs)):
            calc_id = f"calc_{i}"
            if calc_id not in self.nodes:
                continue
            calc = self.nodes[calc_id]

            if calc.expression is None:
                continue

            # Extract numbers from expression
            expr_numbers = re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", calc.expression)

            # Check if all numbers in expression are available
            for num_str in expr_numbers:
                num_val = _to_float(num_str)
                # Allow some tolerance for floating point comparison
                is_available = any(abs(num_val - av) < 0.01 for av in available_values)
                if not is_available:
                    # Missing value - graph is invalid
                    return False

        return True

    def _compute_layout(self):
        """Compute positions for nodes in an alternating left-right layout.

        Layout alternates left and right:
        - Inputs at top (y=0), spread horizontally, sorted by first use
        - Calculations alternate left and right as they go down
        """
        # Group nodes by type
        inputs = {k: v for k, v in self.nodes.items() if v.node_type == "input"}
        calcs = {k: v for k, v in self.nodes.items() if v.node_type == "operation"}

        positions = {}
        y_spacing = 1.8
        x_spacing = 2.5
        x_offset = 3.0  # How far left/right from center

        # Sort inputs by the step number where they're first used
        # This reduces edge crossings
        input_first_use = {}
        for input_id in inputs.keys():
            min_step = float("inf")
            for edge in self.edges:
                if edge.from_node == input_id:
                    # Extract step number from calc_N
                    to_node = edge.to_node
                    if to_node.startswith("calc_"):
                        step_num = int(to_node.split("_")[1])
                        min_step = min(min_step, step_num)
            input_first_use[input_id] = min_step if min_step != float("inf") else 0

        # Sort inputs by first use, then by id for stability
        sorted_inputs = sorted(inputs.keys(), key=lambda x: (input_first_use[x], x))

        # Inputs at top (y=0)
        y_current = 0
        num_inputs = len(inputs)
        for i, node_id in enumerate(sorted_inputs):
            x = (i - (num_inputs - 1) / 2) * x_spacing
            positions[node_id] = (x, y_current)

        # Calculations: alternate left and right
        y_current -= y_spacing

        def _calc_index(cid: str) -> int:
            try:
                return int(cid.split("_", 1)[1])
            except Exception:
                return 10**9

        for i, node_id in enumerate(sorted(calcs.keys(), key=_calc_index)):
            # Alternate: even indices go left, odd indices go right
            if i % 2 == 0:
                x = -x_offset  # Left
            else:
                x = x_offset  # Right
            positions[node_id] = (x, y_current)
            y_current -= y_spacing

        return positions

    def visualize(
        self,
        show_question: bool = True,
        figsize: tuple = (12, 8),
        dpi: int = 150,
        minimal: bool = False,
    ):
        """Create a matplotlib visualization of the computational graph.

        Args:
            show_question: Whether to include the question text as a title
            figsize: Figure size as (width, height) tuple
            dpi: DPI for the output image
            minimal: If True, render without text labels (just colored shapes)

        Returns:
            matplotlib Figure and Axes objects
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Compute layout
        positions = self._compute_layout()

        # Normalize positions for minimal mode to ensure consistent sizing
        if minimal and positions:
            # Get current bounds
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]

            if xs and ys:
                # Find current extent
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_range = x_max - x_min if x_max > x_min else 1.0
                y_range = y_max - y_min if y_max > y_min else 1.0

                # Scale to fit within a balanced range for minimal mode
                target_range = 10.0  # Balanced spacing to avoid overlap
                scale = min(target_range / x_range, target_range / y_range)

                # Center and scale all positions
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                positions = {
                    node_id: ((x - x_center) * scale, (y - y_center) * scale)
                    for node_id, (x, y) in positions.items()
                }

        # Draw edges first (so they appear behind nodes)
        for edge in self.edges:
            if edge.from_node in positions and edge.to_node in positions:
                x1, y1 = positions[edge.from_node]
                x2, y2 = positions[edge.to_node]

                # Draw arrow - much bolder and darker in minimal mode
                if minimal:
                    arrow = FancyArrowPatch(
                        (x1, y1 - 0.25),
                        (x2, y2 + 0.25),
                        arrowstyle="->",
                        mutation_scale=40,
                        linewidth=10.0,
                        color="#000000",
                        alpha=1.0,
                        zorder=1,
                    )
                else:
                    arrow = FancyArrowPatch(
                        (x1, y1 - 0.25),
                        (x2, y2 + 0.25),
                        arrowstyle="->",
                        mutation_scale=20,
                        linewidth=2.5,
                        color="#333333",
                        alpha=0.9,
                        zorder=1,
                    )
                ax.add_patch(arrow)

        # Draw nodes
        for node_id, node in self.nodes.items():
            if node_id not in positions:
                continue

            x, y = positions[node_id]

            # Determine node appearance - all nodes are same-size squares
            size = 1.6 if minimal else 1.2  # Bigger squares in minimal mode
            width, height = size, size

            if node.node_type == "input":
                color = "#64B5F6"  # bright blue
                label = f"{node.value}"
            else:  # operation
                # Color by operation type - bright colors
                expr = node.expression or ""
                if "+" in expr:
                    color = "#66BB6A"  # bright green (addition)
                elif "-" in expr:
                    color = "#EF5350"  # bright red (subtraction)
                elif "*" in expr or "×" in expr:
                    color = "#AB47BC"  # bright purple (multiplication)
                elif "/" in expr or "÷" in expr:
                    color = "#FFEE58"  # bright yellow (division)
                else:
                    color = "#BDBDBD"  # gray (unknown)
                label = f"{node.expression}\n= {node.value}"

            # Draw box
            box = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.1",
                edgecolor="black",
                facecolor=color,
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(box)

            # Add text (skip in minimal mode)
            if not minimal:
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    zorder=3,
                )

        # Set axis limits with padding and equal aspect ratio
        if positions:
            if minimal:
                # Fixed canvas bounds for consistent node sizes
                ax.set_xlim(-8, 8)
                ax.set_ylim(-8, 8)
                ax.set_aspect("equal", adjustable="box")
            else:
                xs = [p[0] for p in positions.values()]
                ys = [p[1] for p in positions.values()]
                x_margin = 2
                y_margin = 1
                ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
                ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
                ax.set_aspect("equal", adjustable="box")

        # Add title with question and answer if requested (skip in minimal mode)
        if show_question and not minimal:
            # Wrap text every 60 characters
            def wrap_text(text, width=60):
                words = text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    word_length = len(word)
                    if current_length + word_length + len(current_line) <= width:
                        current_line.append(word)
                        current_length += word_length
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = word_length

                if current_line:
                    lines.append(" ".join(current_line))

                return "\n".join(lines)

            wrapped_question = wrap_text(self.question, width=60)
            wrapped_answer = wrap_text(self.answer, width=60)
            full_text = f"Question: {wrapped_question}\n\nSolution: {wrapped_answer}"
            ax.set_title(full_text, fontsize=8, pad=20, loc="left")

        # Draw X if graph is invalid (gray in minimal mode, red otherwise)
        if not self.is_valid():
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Choose color based on mode
            x_color = (
                "#666666" if minimal else "red"
            )  # Darker gray in minimal, red otherwise
            x_alpha = 0.55 if minimal else 0.8
            x_linewidth = (
                32 if minimal else 14
            )  # Much bolder, especially in minimal mode
            outline_alpha = 0.75 if minimal else 0.9
            outline_lw = x_linewidth + (10 if minimal else 6)

            # Draw two diagonal lines forming an X, with a black outline so it pops.
            base_kwargs = {"zorder": 100, "solid_capstyle": "round"}
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[0], ylim[1]],
                color="black",
                linewidth=outline_lw,
                alpha=outline_alpha,
                zorder=99,
                solid_capstyle="round",
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[0], ylim[1]],
                color=x_color,
                linewidth=x_linewidth,
                alpha=x_alpha,
                **base_kwargs,
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[1], ylim[0]],
                color="black",
                linewidth=outline_lw,
                alpha=outline_alpha,
                zorder=99,
                solid_capstyle="round",
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[1], ylim[0]],
                color=x_color,
                linewidth=x_linewidth,
                alpha=x_alpha,
                **base_kwargs,
            )
            # Add "INVALID" text (only in non-minimal mode)
            if not minimal:
                mid_x = (xlim[0] + xlim[1]) / 2
                mid_y = (ylim[0] + ylim[1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    "INVALID GRAPH\n(Missing Steps)",
                    fontsize=20,
                    fontweight="bold",
                    color="red",
                    ha="center",
                    va="center",
                    zorder=101,
                    bbox={
                        "boxstyle": "round,pad=0.5",
                        "facecolor": "white",
                        "edgecolor": "red",
                        "linewidth": 3,
                        "alpha": 0.9,
                    },
                )

        ax.axis("off")
        plt.tight_layout()

        return fig, ax

    def render(
        self,
        output_path: str,
        show_question: bool = True,
        format: str = "png",
        figsize: tuple = (10, 8),
        dpi: int = 150,
        minimal: bool = False,
        view: bool = False,
    ) -> str:
        """Render the graph to a file using matplotlib.

        Args:
            output_path: Path to save the graph (without extension)
            show_question: Whether to include the question text
            format: Output format (png, svg, pdf, jpg)
            figsize: Figure size as (width, height) tuple
            dpi: DPI for the output image
            minimal: If True, render without text labels
            view: If True, open the rendered file in a browser/viewer

        Returns:
            Full path to the saved file
        """
        fig, ax = self.visualize(
            show_question=show_question, figsize=figsize, dpi=dpi, minimal=minimal
        )

        # Save figure
        output_file = f"{output_path}.{format}"
        fig.savefig(output_file, format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if view:
            webbrowser.open(f"file://{output_file}")

        return output_file

    def to_dict(self) -> dict[str, Any]:
        """Export graph structure as a dictionary.

        Returns:
            Dictionary with nodes, edges, and metadata
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "final_answer": self.final_answer,
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "value": node.value,
                    "expression": node.expression,
                    "description": node.description,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {"from": edge.from_node, "to": edge.to_node, "label": edge.label}
                for edge in self.edges
            ],
        }


def parse_gsm8k_document(document: str | dict) -> tuple[str, str]:
    """Extract question and answer from a GSM8K document.

    Args:
        document: Either a string or dict with 'document' key

    Returns:
        Tuple of (question, answer)
    """
    if isinstance(document, dict):
        text = document.get("document", "")
    else:
        text = document

    # Split on "Answer:" to separate question and answer
    parts = text.split("\nAnswer:", 1)
    if len(parts) != 2:
        parts = text.split("Answer:", 1)

    if len(parts) == 2:
        question = parts[0].replace("Question:", "").strip()
        answer = parts[1].strip()
        return question, answer
    # Fallback: treat whole thing as question
    return text.strip(), ""


def visualize_gsm8k_problem(
    document: str | dict,
    output_path: str,
    show_question: bool = True,
    format: str = "png",
    view: bool = False,
    minimal: bool = False,
    figsize: tuple = (10, 8),
    dpi: int = 150,
) -> str:
    """Convenience function to parse and visualize a GSM8K problem.

    Args:
        document: GSM8K document (string or dict)
        output_path: Path to save the visualization (without extension)
        show_question: Whether to include the question text
        format: Output format (png, svg, pdf, etc.)
        view: Whether to open the file after rendering
        minimal: If True, render without text labels
        figsize: Figure size as (width, height) tuple
        dpi: DPI for the output image

    Returns:
        Path to the generated visualization file
    """
    question, answer = parse_gsm8k_document(document)
    graph = GSM8KComputationalGraph(question, answer)
    graph.parse()
    return graph.render(
        output_path,
        show_question=show_question,
        format=format,
        view=view,
        minimal=minimal,
        figsize=figsize,
        dpi=dpi,
    )
