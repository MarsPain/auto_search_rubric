#!/bin/bash
# Example: Run auto_search_rubric with content extraction for call summary dataset
# This demonstrates extracting <通话内容> tags before passing to verifier

set -e

echo "=========================================="
echo "Running with content extraction enabled"
echo "=========================================="

# Mock backend demo (no API key needed)
echo ""
echo "1. Mock backend with tag content extraction:"
echo "----------------------------------------"
python3 -m autosr.cli \
    --dataset examples/demo_dataset.json \
    --mode evolutionary \
    --backend mock \
    --extract-strategy tag \
    --extract-tag "content" \
    --generations 3 \
    --population-size 4 \
    --output artifacts/demo_with_extraction.json \
    --verbose

echo ""
echo "=========================================="
echo "Content extraction demo completed"
echo "Output saved to: artifacts/demo_with_extraction.json"
echo "=========================================="

# Show help for content extraction options
echo ""
echo "Content extraction CLI options:"
echo "  --extract-strategy {tag,regex,identity}   Extraction strategy (default: identity)"
echo "  --extract-tag NAME                        Tag name for 'tag' strategy"
echo "  --extract-pattern PATTERN                 Regex pattern for 'regex' strategy"
echo "  --extract-join-separator SEP              Separator for multiple matches"
echo ""
echo "Examples:"
echo "  # Extract from XML-style tags"
echo "  --extract-strategy tag --extract-tag 'content'"
echo ""
echo "  # Extract using regex pattern"
echo "  --extract-strategy regex --extract-pattern 'Data:\\s*(.+)'"
echo ""
echo "  # No extraction (default behavior)"
echo "  --extract-strategy identity"
