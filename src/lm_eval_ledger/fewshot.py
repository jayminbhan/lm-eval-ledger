# fewshot.py
"""Few-shot prompt-block and chat-message builders for all eval modes."""
from __future__ import annotations

from pathlib import Path

from .tasks.base import load_jsonl


def build_fewshot_block(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block from curated few-shot file or training data.

    Uses task.build_prompt to format each example properly (includes MCQ choices
    for MMLU, proper question format for other tasks), then appends the answer.

    Supports two formats:
    1. Simple answer-only: fewshot_answer_field contains the complete answer
    2. Solution+Answer: fewshot_solution_field has reasoning, fewshot_final_answer_field has the answer
       Format: {solution} \boxed{answer}
    """
    # Format instruction - always included even for 0-shot
    format_instruction = r"Output format: end your response with \boxed{<answer>} where <answer> is the final answer."

    if k <= 0:
        return format_instruction

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        print(f"[WARN] No few-shot examples available for {task.name}; "
              f"falling back to 0-shot prompt")
        return format_instruction

    blocks = []

    # Add format instruction at the start
    blocks.append(format_instruction)

    # Check which format to use
    use_solution_answer_format = bool(task.fewshot_solution_field)

    for ex in examples:
        # Use task's build_prompt to format properly (includes MCQ choices, etc.)
        question_part = task.build_prompt(ex, "")

        if use_solution_answer_format:
            # Solution + Answer format: {reasoning} \boxed{answer}
            solution = ex.get(task.fewshot_solution_field, "")
            # Handle list values (e.g., OlympiadBench solution field is a list)
            if isinstance(solution, list):
                solution = solution[0] if solution else ""
            solution = str(solution).strip()

            # Get final answer if available
            final_answer = ""
            if task.fewshot_final_answer_field:
                final_answer = ex.get(task.fewshot_final_answer_field, "")
                if isinstance(final_answer, list):
                    final_answer = final_answer[0] if final_answer else ""
                final_answer = str(final_answer).strip() if final_answer else ""

            if question_part and solution:
                if final_answer:
                    # Full format with reasoning and answer
                    blocks.append(f"{question_part}\n{solution}\n\n\\boxed{{{final_answer}}}")
                else:
                    # Proof-type problems without numeric answer (just solution)
                    blocks.append(f"{question_part}\n{solution}")
        else:
            # Simple answer-only format (e.g., GSM8K where answer contains "solution text\n#### number")
            answer = ex.get(task.fewshot_answer_field, "")
            # Handle list values
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()
            if question_part and answer:
                # Check if answer contains #### delimiter (like GSM8K format)
                if "####" in answer:
                    # Split into solution and final answer
                    parts = answer.split("####")
                    solution_part = parts[0].strip()
                    final_answer = parts[-1].strip()
                    # Don't double-wrap if solution already contains \boxed{}
                    if "\\boxed{" in solution_part:
                        blocks.append(f"{question_part}\n{solution_part}")
                    else:
                        blocks.append(f"{question_part}\n{solution_part}\n\n\\boxed{{{final_answer}}}")
                else:
                    # Plain answer without solution
                    blocks.append(f"{question_part}\n\\boxed{{{answer}}}")

    return "\n\n".join(blocks)


def build_fewshot_block_logprob_token(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block for first-token log-probability MCQ evaluation.

    Simpler than generate mode: no format instruction, no \\boxed{}.
    Each few-shot example shows the question followed by the gold answer letter.
    """
    if k <= 0:
        return ""

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return ""

    blocks = []

    for ex in examples:
        # Build question prompt (e.g., "Question: ...\nA. ...\nB. ...\nAnswer:")
        question_part = task.build_prompt(ex, "")
        # Get the gold answer letter (e.g., "A", "B", "C", "D")
        gold = task.extract_gold(ex)
        if question_part and gold:
            blocks.append(f"{question_part} {gold}")

    return "\n\n".join(blocks)


def build_fewshot_block_logprob_seq(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block for completion log-likelihood MCQ evaluation.

    Each few-shot example shows the question followed by the full answer text
    (not the letter), consistent with the evaluation that scores full answer texts.
    """
    if k <= 0:
        return ""

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return ""

    blocks = []

    for ex in examples:
        question_part = task.build_prompt(ex, "")
        gold_label = task.extract_gold(ex)
        # Map gold label to full answer text
        if task.get_choice_texts and task.choice_labels:
            choice_texts = task.get_choice_texts(ex)
            if gold_label in task.choice_labels:
                gold_idx = task.choice_labels.index(gold_label)
                if gold_idx < len(choice_texts):
                    answer_text = choice_texts[gold_idx]
                    if question_part and answer_text:
                        blocks.append(f"{question_part} {answer_text}")
                    continue
        # Fallback: use the label itself
        if question_part and gold_label:
            blocks.append(f"{question_part} {gold_label}")

    return "\n\n".join(blocks)


# ======================================================
# MULTI-TURN CHAT FEW-SHOT (for instruct/chat models)
# ======================================================

def _load_fewshot_raw(data_dir: Path, task, k: int) -> list[dict]:
    """Load few-shot examples from local file or HF. Returns [] if nothing configured."""
    if k <= 0:
        return []

    # Prefer local fewshot file if configured
    if task.fewshot_path:
        local_path = data_dir / task.fewshot_path
        if local_path.exists():
            examples = load_jsonl(local_path)
            print(f"[INFO] Loaded {len(examples)} few-shot examples from {local_path}")
            return examples[:k]

    if task.hf_repo and task.hf_fewshot_split:
        from datasets import load_dataset
        config = task.hf_fewshot_config or task.hf_config
        ds = load_dataset(task.hf_repo, config, split=task.hf_fewshot_split)
        return [dict(row) for row in ds][:k]

    return []


def build_fewshot_chat_messages(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for generate eval mode.

    Returns list of message dicts (system + user/assistant pairs) or None if no examples.
    The caller should append the final test question as a user message.
    """
    format_instruction = r"Output format: end your response with \boxed{<answer>} where <answer> is the final answer."

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = [{"role": "system", "content": format_instruction}]
    use_solution_answer_format = bool(task.fewshot_solution_field)

    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        if not user_msg:
            continue

        if use_solution_answer_format:
            solution = ex.get(task.fewshot_solution_field, "")
            if isinstance(solution, list):
                solution = solution[0] if solution else ""
            solution = str(solution).strip()
            if not solution:
                continue

            final_answer = ""
            if task.fewshot_final_answer_field:
                final_answer = ex.get(task.fewshot_final_answer_field, "")
                if isinstance(final_answer, list):
                    final_answer = final_answer[0] if final_answer else ""
                final_answer = str(final_answer).strip() if final_answer else ""

            if final_answer:
                assistant_msg = f"{solution}\n\n\\boxed{{{final_answer}}}"
            else:
                assistant_msg = solution
        else:
            answer = ex.get(task.fewshot_answer_field, "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()
            if not answer:
                continue

            if "####" in answer:
                parts = answer.split("####")
                solution_part = parts[0].strip()
                final_answer = parts[-1].strip()
                if "\\boxed{" in solution_part:
                    assistant_msg = solution_part
                else:
                    assistant_msg = f"{solution_part}\n\n\\boxed{{{final_answer}}}"
            else:
                assistant_msg = f"\\boxed{{{answer}}}"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Only return if we have actual example pairs (not just system message)
    return messages if len(messages) > 1 else None


def build_fewshot_chat_messages_logprob_token(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for logprob_token eval mode.

    Returns list of message dicts (user/assistant pairs) or None if no examples.
    """
    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = []
    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        gold = task.extract_gold(ex)
        if user_msg and gold:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": gold})

    return messages if messages else None


def build_fewshot_chat_messages_logprob_seq(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for logprob_seq eval mode.

    Returns list of message dicts (user/assistant pairs) or None if no examples.
    """
    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = []
    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        gold_label = task.extract_gold(ex)

        answer_text = gold_label  # fallback
        if task.get_choice_texts and task.choice_labels:
            choice_texts = task.get_choice_texts(ex)
            if gold_label in task.choice_labels:
                gold_idx = task.choice_labels.index(gold_label)
                if gold_idx < len(choice_texts):
                    answer_text = choice_texts[gold_idx]

        if user_msg and answer_text:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": answer_text})

    return messages if messages else None
