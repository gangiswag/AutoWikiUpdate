import difflib
import re
import wikitextparser as wtp



def finegrained_diff(old_text, new_text):

    diff = difflib.unified_diff(
                            re.split(r'(?<=[.!?;])\s+', old_text),
                            re.split(r'(?<=[.!?;])\s+', new_text),
                            lineterm='',
                            fromfile='old_text',
                            tofile='new_text')
    
    changes = []
    non_trivial_edits = [line for line in diff if re.search(r'[a-zA-Z]', line)]
    for line in non_trivial_edits:
        if line.startswith('+') and not line.startswith('+++'):
            changes.append(["Sentence", "insert", line[1:]])
        elif line.startswith('-') and not line.startswith('---'):
            changes.append(["Sentence", "remove", line[1:]])
    return changes


# Add the function to detect paragraph changes
def finegrained_diff_by_paragraph(old_text, new_text):
    old_paragraphs = re.split(r'\n\s*\n+', old_text.strip())  # Split by double newlines (paragraphs)
    new_paragraphs = re.split(r'\n\s*\n+', new_text.strip())

    diff = difflib.unified_diff(
        old_paragraphs,
        new_paragraphs,
        lineterm='',
        fromfile='old_text',
        tofile='new_text'
    )

    changes = []
    non_trivial_edits = [line for line in diff if re.search(r'[a-zA-Z]', line)]
    
    for line in non_trivial_edits:
        if line.startswith('+') and not line.startswith('+++'):
            changes.append(["Paragraph", "insert", line[1:].strip()])
        elif line.startswith('-') and not line.startswith('---'):
            changes.append(["Paragraph", "remove", line[1:].strip()])
    
    return changes
    
    
