
## Wikipedia Revision Extraction

### Overview


### Extraction

We provide a extensive list of entities for which we can extract historical wikipedia edits ( revisions )

```bash
python3 extract_data.py \
    --entity_list entities.json \
    --dir_path dataset \
    --month 4 \
    --year 2025 \
    --start 0 \
    --end 1 \
```

- `entity_list` path to the json file containing the entity name and wikipedia ID mapping ( entities.json provided for reference)
- `dir_path` the folder to store the data, each entity will be stored a separate json file ( wikipedia_id.json ). Have a look at [Data Format](#data-format) for details on format of storing
- `month` the month upto which to extract the data, note we start from current time and extract back in time.
- `year`the year upto which to extract the data
- `start` start index to extract data for in the list of entites in entity_list
- `end` end index to extract data for in the list of entites in entity_list


### Data Format
Each entity’s revisions are stored in a JSON file named after its Wikipedia page ID, e.g., 123456.json. The structure is a dictionary where each key is a revision timestamp, and its value contains metadata about the revision:

```json
{
  "2025-04-20T18:25:43Z": {
    "revision_id": 123456789,
    "parent_revision_id": 123456788,
    "hierarchy": ["Introduction", "Career", "2020s"],
    "section_information": {
        "1": {
            "title": "Overview",
            "content": "....",
            "sentence_edit": [],
            "paragraph_edit": [],
            "new_urls": [],
            "has_table": [],
            "hierachy_information": {
                "index": "1",
                "line": "Overview",
                "level": 0
            },
            "old_content": ""
        }
  },
  ...
}
```

- `revision_id`: The unique ID of this Wikipedia revision.

- `parent_revision_id`: The ID of the parent revision it is based on.

- `hierarchy`: A list representing the section/subsection structure of the revision.

- `title`: Title of the section

- `content`: The section content in plain text

- `old_content`: The section content in plain text before the edit

- `sentence_edit`: the sections that have been either removed or added

- `paragraph_edit`: the paragraphs that have been either removed or added

- `new_urls`: the urls that have been added
