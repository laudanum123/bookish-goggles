import json

def parse_json_from_string(string):
    start_index = string.find('{')
    end_index = string.rfind('}')
    if start_index == -1 or end_index == -1:
        return None
    print(string)
    print(start_index, end_index)
    json_string = string[start_index:end_index+1]
    #print(json_string)
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError as e:
        print(e)
        return None
