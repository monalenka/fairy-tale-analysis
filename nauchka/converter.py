import re
import json

def simple_jsonl_conversion(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    objects = []
    current = ""
    depth = 0
    
    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
            
        for char in line:
            if char == '{':
                depth += 1
                if depth == 1:
                    if current.strip() and current.strip().startswith('{'):
                        objects.append(current.strip())
                    current = "{"
                else:
                    current += char
            elif char == '}':
                depth -= 1
                current += char
                if depth == 0:
                    objects.append(current)
                    current = ""
            else:
                current += char

    if current.strip() and current.strip().startswith('{'):
        objects.append(current.strip())

    valid_objects = []
    for obj_str in objects:
        try:
            cleaned = obj_str.strip()
            if not cleaned.startswith('{') or not cleaned.endswith('}'):
                continue
                
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            obj = json.loads(cleaned)
            valid_objects.append(obj)
        except Exception as e:
            print(f"Ошибка парсинга: {e}")
            print(f"Проблемный объект: {obj_str[:200]}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in valid_objects:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')


simple_jsonl_conversion('tales.jsonl', 'good_tales.jsonl')