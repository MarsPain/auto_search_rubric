from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from autosr.cli import DEFAULT_MODEL


LLM_API_KEY = os.getenv("LLM_API_KEY")


@unittest.skipUnless(LLM_API_KEY, "LLM_API_KEY is required for integration test")
class TestLLMIntegration(unittest.TestCase):
    def test_cli_runs_iterative_and_evolutionary_with_llm(self) -> None:
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.json"
            iterative_output = tmp_path / "iterative.json"
            evolutionary_output = tmp_path / "evolutionary.json"
            dataset_path.write_text(json.dumps(_tiny_dataset(), ensure_ascii=False), encoding="utf-8")

            iterative_cmd = [
                sys.executable,
                "-m",
                "autosr.cli",
                "--dataset",
                str(dataset_path),
                "--mode",
                "iterative",
                "--output",
                str(iterative_output),
                "--backend",
                "auto",
                "--base-url",
                base_url,
                "--model-default",
                model_name,
                "--llm-max-retries",
                "1",
                "--iterations",
                "1",
            ]
            evolutionary_cmd = [
                sys.executable,
                "-m",
                "autosr.cli",
                "--dataset",
                str(dataset_path),
                "--mode",
                "evolutionary",
                "--output",
                str(evolutionary_output),
                "--backend",
                "auto",
                "--base-url",
                base_url,
                "--model-default",
                model_name,
                "--llm-max-retries",
                "1",
                "--generations",
                "1",
                "--population-size",
                "2",
                "--mutations-per-round",
                "1",
                "--batch-size",
                "1",
            ]

            iterative_result = subprocess.run(
                iterative_cmd,
                check=False,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if iterative_result.returncode != 0:
                self.fail(
                    "iterative CLI failed.\n"
                    f"command: {' '.join(iterative_cmd)}\n"
                    f"exit_code: {iterative_result.returncode}\n"
                    f"stdout:\n{iterative_result.stdout}\n"
                    f"stderr:\n{iterative_result.stderr}"
                )

            evolutionary_result = subprocess.run(
                evolutionary_cmd,
                check=False,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if evolutionary_result.returncode != 0:
                self.fail(
                    "evolutionary CLI failed.\n"
                    f"command: {' '.join(evolutionary_cmd)}\n"
                    f"exit_code: {evolutionary_result.returncode}\n"
                    f"stdout:\n{evolutionary_result.stdout}\n"
                    f"stderr:\n{evolutionary_result.stderr}"
                )

            self.assertIn("Backend: llm", iterative_result.stdout)
            self.assertIn("Backend: llm", evolutionary_result.stdout)
            self.assertTrue(iterative_output.exists())
            self.assertTrue(evolutionary_output.exists())

            iterative_payload = json.loads(iterative_output.read_text(encoding="utf-8"))
            evolutionary_payload = json.loads(evolutionary_output.read_text(encoding="utf-8"))
            self.assertIn("best_rubrics", iterative_payload)
            self.assertIn("best_scores", iterative_payload)
            self.assertIn("best_rubrics", evolutionary_payload)
            self.assertIn("best_scores", evolutionary_payload)


def _tiny_dataset() -> dict[str, object]:
    return {
        "prompts": [
            {
                "prompt_id": "p_smoke",
                "prompt": "Write a short launch email.",
                "candidates": [
                    {
                        "candidate_id": "a",
                        "source": "strong",
                        "text": "Subject: Launch now. Clear value and CTA.",
                    },
                    {
                        "candidate_id": "b",
                        "source": "base",
                        "text": "We launched a product. Please try it.",
                    },
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

# >>> CODEx prompt style >>>
export VIRTUAL_ENV_DISABLE_PROMPT=1
autoload -Uz colors && colors
setopt PROMPT_SUBST
PROMPT='${VIRTUAL_ENV:+"(%F{yellow}"$(basename "$VIRTUAL_ENV")"%f) "}%F{green}%n%f@%F{cyan}%m%f:%F{yellow}%~%f  %F{red}%?%f %F{magenta}SHLVL:${SHLVL}%f %F{blue}%D{%I:%M:%S}%f
# '
# <<< CODEx prompt style <<<

# >>> codex cli tools >>>
# CLI tools: zoxide, eza, ripgrep, fd, less, tmux
if [ -x /opt/homebrew/bin/less ]; then
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
fi

if command -v zoxide >/dev/null 2>&1; then
  eval "$(zoxide init zsh)"
fi

if command -v eza >/dev/null 2>&1; then
  alias ls='eza --group-directories-first --icons=auto'
  alias ll='eza -lh --group-directories-first --icons=auto'
  alias la='eza -lah --group-directories-first --icons=auto'
  alias lt='eza --tree --level=3 --icons=auto'
fi

if command -v fd >/dev/null 2>&1; then
  alias fd-find='fd'
fi

if command -v rg >/dev/null 2>&1; then
  alias rgi='rg -i'
  alias rgf='rg --files'
fi

export LESS='-R -F -X -i'
export LESSHISTFILE='-'
# <<< codex cli tools <<<

# tmux
# alias of tmux
alias tml='tmux ls | less'
alias tmk='tmux kill-session -t'
# attach or new or list
tm() {
  if [ $# -eq 0 ]; then
    echo "Usage: tm <session-name>"
    echo "Existing sessions:"
    tmux ls
    return 1
  fi

  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux attach-session -t "$session"
  else
    tmux new-session -s "$session"
  fi
}

export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/
export ANTHROPIC_API_KEY=sk-kimi-qNoYZo0kwALHdPrxhw1Isma5pvNGL4nxkW0VpsCBLBo77XAe7Sblp65xxDifrQh3
export LLM_API_KEY=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

export PATH="$HOME/.local/bin:$PATH"
