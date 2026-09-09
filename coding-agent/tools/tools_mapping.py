from tools.tools import (
    get_page_content, 
    get_weather, 
    read_file, 
    write_file, 
    run_command
)

tools_mapping = {
    "get_weather": get_weather,
    "get_page_content": get_page_content,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}
