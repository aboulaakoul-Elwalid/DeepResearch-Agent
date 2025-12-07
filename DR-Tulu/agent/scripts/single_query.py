#!/usr/bin/env python3
"""
Single-query CLI for DR-Tulu deep research agent.

This script runs a single research query and outputs structured tags to stdout
for parsing by the gateway. It exits after completing the query.

Output format (line-based for easy parsing):
  [THINK] <thinking text>
  [TOOL_CALL] <tool_name> | <call_id>
  [TOOL_OUTPUT] <output text>
  [ANSWER] <answer text>
  [DONE]

Usage:
    python scripts/single_query.py --config workflows/auto_search_deep.yaml --query "What is quantum computing?"
"""

import asyncio
import sys
import json
import re
from pathlib import Path
from typing import Optional

# Import the workflow
sys.path.insert(0, str(Path(__file__).parent.parent))
from workflows.auto_search_sft import AutoReasonSearchWorkflow
from dr_agent.tool_interface.data_types import DocumentToolOutput


def clean_text(t: str) -> str:
    """Reduce multiple newlines and clean text for output."""
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def looks_like_final_answer(text: str, verbose: bool = False) -> bool:
    """Heuristic to catch full answers even when <answer> tags are missing.

    This is a CONSERVATIVE heuristic - only returns True when we're confident.
    """
    if not text:
        return False

    # Strip tool/thinking tags to get actual content
    stripped = re.sub(r"<call_tool[^>]*>.*?</call_tool>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<tool_output>.*?</tool_output>", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    if verbose:
        print(
            f"[DEBUG] looks_like_final_answer: stripped length = {len(stripped)}",
            file=sys.stderr,
        )

    # Must have substantial content
    if len(stripped) < 500:
        return False

    # Strong signal: citations indicate synthesized answer
    if "<cite id=" in stripped:
        return True

    # Strong signal: explicit conclusion markers
    conclusion_markers = [
        "in conclusion",
        "to summarize",
        "in summary",
        "## conclusion",
        "## summary",
    ]
    normalized = stripped.lower()
    for marker in conclusion_markers:
        if marker in normalized:
            if verbose:
                print(
                    f"[DEBUG] looks_like_final_answer: found '{marker}'",
                    file=sys.stderr,
                )
            return True

    # Don't trigger on length alone - too many false positives
    return False


def escape_for_output(text: str) -> str:
    """Escape text for line-based output (replace newlines with \\n)."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def unescape_output(text: str) -> str:
    """Unescape text from line-based output."""
    return text.replace("\\n", "\n").replace("\\\\", "\\")


async def run_single_query(
    workflow: AutoReasonSearchWorkflow,
    query: str,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
):
    """Run a single query and output structured results to stdout."""

    # State tracking
    last_text_len = 0
    in_answer = False
    answer_complete = False  # Track if we've finished sending an answer
    current_thinking = ""
    current_answer = ""
    cumulative_response = (
        ""  # Track all assistant generations to detect untagged answers
    )

    def flush_output():
        """Flush stdout to ensure real-time output."""
        sys.stdout.flush()

    def output_think(text: str):
        """Output thinking text."""
        if text.strip():
            # Clean UI hallucination patterns
            text = re.sub(r"Thought for \d+ seconds?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"Phase \d+:", "", text)  # Remove repeated phase markers
            text = text.strip()
            if text:
                print(f"[THINK] {escape_for_output(text)}")
                flush_output()

    def output_tool_call(tool_name: str, call_id: str, args: str = ""):
        """Output tool call."""
        print(f"[TOOL_CALL] {tool_name} | {call_id} | {escape_for_output(args)}")
        flush_output()

    def output_tool_result(call_id: str, output: str):
        """Output tool result."""
        # Truncate very long outputs
        if len(output) > 2000:
            output = output[:2000] + "... [truncated]"
        print(f"[TOOL_OUTPUT] {call_id} | {escape_for_output(output)}")
        flush_output()

    def output_answer(text: str):
        """Output answer text."""
        # Clean UI hallucination patterns from answer
        text = re.sub(r"Thought for \d+ seconds?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Phase \d+:", "", text)  # Remove repeated phase markers
        if text.strip():
            print(f"[ANSWER] {escape_for_output(text)}")
            flush_output()

    def step_callback(text: str, tool_calls):
        """Callback for each step of the workflow."""
        nonlocal \
            last_text_len, \
            in_answer, \
            answer_complete, \
            current_thinking, \
            current_answer, \
            cumulative_response

        # If we've already completed an answer, ignore further callbacks
        # This prevents looping when the model generates duplicate answers
        if answer_complete:
            return

        # Handle text stream reset (new generation)
        if text and len(text) < last_text_len:
            # Only reset if we haven't completed an answer yet
            if not answer_complete:
                last_text_len = 0
                # Don't reset in_answer - preserve answer state
                current_thinking = ""

        # Process new text
        if text and len(text) > last_text_len:
            new_chunk = text[last_text_len:]
            last_text_len = len(text)
            cumulative_response += new_chunk

            # Check for answer tag
            if "<answer>" in text and not in_answer:
                # Split at answer tag
                parts = text.split("<answer>", 1)
                thinking_part = parts[0]
                answer_part = (
                    parts[1].replace("</answer>", "") if len(parts) > 1 else ""
                )

                # Check if answer is complete (has closing tag)
                if "</answer>" in text:
                    answer_complete = True

                # Output remaining thinking if any
                if thinking_part and thinking_part != current_thinking:
                    # Remove think tags for cleaner output
                    clean_thinking = thinking_part.replace("<think>", "").replace(
                        "</think>", ""
                    )
                    output_think(clean_thinking)
                    current_thinking = thinking_part

                in_answer = True
                current_answer = answer_part
                output_answer(answer_part)
            elif in_answer:
                # Continue streaming answer
                answer_chunk = new_chunk.replace("</answer>", "")

                # Check if this chunk contains the closing tag
                if "</answer>" in new_chunk:
                    answer_complete = True

                current_answer += answer_chunk
                output_answer(answer_chunk)
            else:
                # Streaming thinking
                current_thinking += new_chunk
                # Don't output every chunk of thinking - batch it
                # We'll output on tool calls or when answer starts

                # Heuristic: if the model is dumping a full report without <answer> tags, treat it as the answer
                if looks_like_final_answer(cumulative_response, verbose=verbose):
                    in_answer = True
                    answer_complete = True
                    current_answer = cumulative_response
                    output_answer(cumulative_response)
                    current_thinking = ""

        # Handle tool calls
        if tool_calls:
            # Output any pending thinking before tool calls
            if current_thinking and not in_answer:
                clean_thinking = current_thinking.replace("<think>", "").replace(
                    "</think>", ""
                )
                clean_thinking = clean_thinking.replace("<answer>", "").replace(
                    "</answer>", ""
                )
                output_think(clean_thinking)
                current_thinking = ""

            for tool_call in tool_calls:
                tool_name = getattr(tool_call, "tool_name", "unknown")
                call_id = getattr(tool_call, "call_id", "call_0")

                # Get arguments/query
                args = ""
                if hasattr(tool_call, "arguments"):
                    args = str(tool_call.arguments)

                output_tool_call(tool_name, call_id, args)

                # Output tool result
                if isinstance(tool_call, DocumentToolOutput) and tool_call.documents:
                    docs_output = []
                    for idx, doc in enumerate(tool_call.documents):
                        doc_str = (
                            doc.stringify() if hasattr(doc, "stringify") else str(doc)
                        )
                        docs_output.append(f"[{idx}] {doc_str}")
                    output_tool_result(call_id, "\n".join(docs_output))
                elif hasattr(tool_call, "output") and tool_call.output:
                    output_tool_result(call_id, str(tool_call.output))

            # Reset text position for next iteration, but preserve answer state
            # IMPORTANT: Don't reset in_answer - once we've started answering,
            # we should continue in answer mode to prevent duplicate answer loops
            last_text_len = 0
            current_thinking = ""
            # Keep current_answer to detect duplicates

    # Run the workflow
    try:
        result = await workflow(
            problem=query,
            dataset_name=dataset_name,
            verbose=verbose,
            step_callback=step_callback,
        )

        # Output final answer if not already done
        generated_text = result.get("generated_text", "")

        if generated_text:
            if "<answer>" in generated_text:
                # Extract final answer from <answer> tags
                answer_match = re.search(
                    r"<answer>(.*?)</answer>", generated_text, re.DOTALL
                )
                if answer_match and not current_answer:
                    output_answer(answer_match.group(1))
            elif not current_answer:
                # No <answer> tags found - extract the final response as the answer
                # This handles models that don't use proper answer tags
                # Try to find the last substantial text after tool outputs

                # Split by common markers to find the final response
                text_parts = re.split(
                    r"</tool_output>|</snippet>|</webpage>", generated_text
                )
                if text_parts:
                    final_part = text_parts[-1].strip()
                    # Clean up any remaining tags
                    final_part = re.sub(r"</?think>", "", final_part)
                    final_part = re.sub(
                        r"<call_tool[^>]*>.*?</call_tool>",
                        "",
                        final_part,
                        flags=re.DOTALL,
                    )
                    final_part = final_part.strip()

                    if final_part and len(final_part) > 50:
                        # We have a substantial final response
                        output_answer(final_part)
                    elif generated_text.strip():
                        # Fallback: output the entire generated text
                        # Remove tool call patterns and thinking tags
                        cleaned = re.sub(
                            r"<think>.*?</think>", "", generated_text, flags=re.DOTALL
                        )
                        cleaned = re.sub(
                            r"<call_tool[^>]*>.*?</call_tool>",
                            "",
                            cleaned,
                            flags=re.DOTALL,
                        )
                        cleaned = re.sub(
                            r"<tool_output>.*?</tool_output>",
                            "",
                            cleaned,
                            flags=re.DOTALL,
                        )
                        cleaned = cleaned.strip()
                        if cleaned:
                            output_answer(cleaned)

        # Output completion
        print("[DONE]")
        flush_output()

        # Return result for potential further processing
        return result

    except Exception as e:
        print(f"[ERROR] {escape_for_output(str(e))}")
        flush_output()
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single deep research query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to workflow configuration YAML file",
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="The research query to answer",
    )
    parser.add_argument(
        "--dataset-name",
        "-d",
        default="long_form",
        help="Dataset name for prompt configuration (default: long_form)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--config-overrides",
        help="Override config params: 'param1=value1,param2=value2'",
    )

    args = parser.parse_args()

    # Parse config overrides
    overrides = {}
    if args.config_overrides:
        for pair in args.config_overrides.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            if value.lower() == "true":
                overrides[key] = True
            elif value.lower() == "false":
                overrides[key] = False
            elif value.lower() in ["none", "null"]:
                overrides[key] = None
            elif value.isdigit():
                overrides[key] = int(value)
            else:
                try:
                    overrides[key] = float(value)
                except ValueError:
                    overrides[key] = value

    # Create workflow
    try:
        workflow = AutoReasonSearchWorkflow(configuration=args.config, **overrides)
    except Exception as e:
        print(f"[ERROR] Failed to create workflow: {e}")
        sys.exit(1)

    # Run query
    try:
        asyncio.run(
            run_single_query(
                workflow=workflow,
                query=args.query,
                dataset_name=args.dataset_name,
                verbose=args.verbose,
            )
        )
    except KeyboardInterrupt:
        print("[ERROR] Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
