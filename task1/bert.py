from transformers import BertTokenizer, BertModel


def main():
    tokenizer = BertTokenizer('task1/models/RuBERT_conversational/vocab.txt')
    model = BertModel.from_pretrained('task1/models/RuBERT_conversational')

    encoded_input = tokenizer('Мама мыла раму', return_tensors='pt')
    output = model(**encoded_input)

    print('input_ids:', encoded_input['input_ids'])

    for key, value in output.items():
        print(key, value.shape)


if __name__ == '__main__':
    main()
