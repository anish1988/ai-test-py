from    llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "note.txt")


def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")
    with open(note_file, "a") as f:
        f.write([note + "\n"])

    return f"Note saved: {note}"
    # return f"Note saved: {note}"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_engine",
    description="this tool can save a text based note to a file for the user.",

)    