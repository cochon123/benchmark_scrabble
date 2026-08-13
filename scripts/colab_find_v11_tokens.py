from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
stable = []
for codepoint in range(0x4E00, 0xA000):
    token = chr(codepoint)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1 or tokenizer.decode(encoded) != token:
        continue
    token_id = encoded[0]
    contexts = (f"{token}:.", f" {token}:.", f"AT|{token}|A")
    if all(tokenizer.encode(context, add_special_tokens=False).count(token_id) == 1 for context in contexts):
        stable.append(token)
    if len(stable) == 225:
        break
print("COUNT", len(stable))
print("".join(stable))
