import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import redditcleaner
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.EMOJI)

all_cats = ['incivility',
            'harassment',
            'spam',
            'format',
            'content',
            'off-topic',
            'hatespeech',
            'trolling',
            'meta-rules'
            ]


class NormVioSeq(Dataset):
    def __init__(self,
                 phase,
                 n_few_shot=0,
                 cat='all',
                 comm='all',
                 max_context_size=5,
                 max_n_tokens=128,
                 n_workers=4,
                 seed=2022):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=n_workers, progress_bar=False)

        df = pd.read_csv(f'data/{phase}.csv', converters={'context': eval})
        if cat != 'all':
            if cat[0] != '~':
                mask = df['cats'].apply(lambda x: cat in x)
            else:
                mask = df['cats'].apply(lambda x: cat[1:] not in x)
            df = df.loc[mask]
        if comm != 'all':
            if comm[0] != '~':
                mask = df['subreddit'].apply(lambda x: comm == x)
            else:
                mask = df['subreddit'].apply(lambda x: comm[1:] != x)
            df = df.loc[mask]

        if n_few_shot > 0 and phase != 'test':
            df_few_shot = []
            for each in all_cats:
                df_sub = df[df['cats'].apply(lambda x: each in x)]
                df_sub = df_sub.sample(n=n_few_shot, random_state=seed)
                df_few_shot.append(df_sub)
            df_few_shot = pd.concat(df_few_shot)
            df = df_few_shot

        n = df.shape[0]
        print(f'**********{phase} set, {n} comments**********')

        def truncate_context(x):
            # only keep a few predecessors
            x = x[-max_context_size:]
            return x
        df['context'] = df['context'].parallel_apply(truncate_context)

        def reddit_clean(x):
            return p.tokenize(redditcleaner.clean(x))

        def reddit_batch_clean(x):
            return [p.tokenize(redditcleaner.clean(e)) for e in x]

        def augment_comment(row):
            comment = row['final_comment']
            subrredit = row['subreddit']
            rule_text = row['rule_texts']
            return f'subrreddit: r/{subrredit}. rule_text: {rule_text}. comment: {comment}.'

        def augment_context(row):
            context = row['context']
            subrredit = row['subreddit']
            rule_text = row['rule_texts']
            return [f'subrreddit: r/{subrredit}. rule_text: {rule_text}. comment: {comment}.' for comment in context]

        subreddits = df['subreddit'].tolist()
        df['final_comment'] = df['final_comment'].parallel_apply(reddit_clean)
        df['context'] = df['context'].parallel_apply(reddit_batch_clean)
        comments = df.parallel_apply(augment_comment, axis=1)
        contexts = df.parallel_apply(augment_context, axis=1)
        conversations = [x + [y] for x, y in zip(contexts, comments)]
        # rule_texts = df['rule_texts'].tolist()
        for cat in all_cats:
            n_cat = df['cats'].apply(lambda x: cat in x).sum()
            print(f'{cat}: {n_cat / n:.2f}')
        print()
        cats = df['cats'].tolist()
        labels = df['bool_derail'].astype(int).tolist()
        conv_lens = pd.Series(conversations).apply(len).tolist()

        conversations_1d = list(itertools.chain(*conversations))

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        # encode the conversations
        print('Tokenizing....')
        encodings = tokenizer(conversations_1d, padding='max_length', truncation=True, max_length=max_n_tokens,
                              return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        print('Done\n')

        def slice_list(lst, chunk_sizes):
            result = []
            i = 0
            for size in chunk_sizes:
                result.append(lst[i:i + size])
                i += size
            return result

        input_ids = slice_list(input_ids, conv_lens)
        attention_mask = slice_list(attention_mask, conv_lens)

        # pad the conversations
        dummpy_input_ids = torch.tensor(
            [tokenizer.pad_token_id, tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_n_tokens - 2))
        dummpy_attention_mask = torch.tensor([1, 1] + [0] * (max_n_tokens - 2))

        def pad_conv(i):
            conv_len = conv_lens[i]
            n_padding = max_context_size + 1 - conv_len
            input_ids[i] = torch.cat([input_ids[i], dummpy_input_ids.repeat(n_padding, 1)], dim=0)
            attention_mask[i] = torch.cat([attention_mask[i], dummpy_attention_mask.repeat(n_padding, 1)], dim=0)

        indices = pd.Series(range(n))
        indices.apply(pad_conv)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.subreddits = subreddits
        self.conv_lens = conv_lens
        self.cats = cats
        self.labels = labels

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'subreddit': self.subreddits[index],
            'conv_len': self.conv_lens[index],
            'cat': self.cats[index],
            'label': self.labels[index]
        }
        return item

    def __len__(self):
        return len(self.labels)


def create_normvio_prompt_dataset(phase,
                                  n_few_shot=0,
                                  cat='all',
                                  comm='all',
                                  max_context_size=5,
                                  seed=2022
                                  ):
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=4)

    df = pd.read_csv(f'data/{phase}.csv', converters={'context': eval})
    if cat != 'all':
        if cat[0] != '~':
            mask = df['cats'].apply(lambda x: cat in x)
        else:
            mask = df['cats'].apply(lambda x: cat[1:] not in x)
        df = df.loc[mask]
    if comm != 'all':
        if comm[0] != '~':
            mask = df['subreddit'].apply(lambda x: comm == x)
        else:
            mask = df['subreddit'].apply(lambda x: comm[1:] != x)
        df = df.loc[mask]

    if n_few_shot > 0 and phase != 'test':
        df_few_shot = []
        for each in all_cats:
            df_sub = df[df['cats'].apply(lambda x: each in x)]
            df_sub = df_sub.sample(n=n_few_shot, random_state=seed)
            df_few_shot.append(df_sub)
        df_few_shot = pd.concat(df_few_shot)
        df = df_few_shot

    n = df.shape[0]
    print(f'**********{phase} set, {n} comments**********\n')

    def truncate_context(x):
        n = len(x)
        if n >= max_context_size:
            x = x[-max_context_size:]
        else:
            x = ['None.'] * (max_context_size - n) + x
        return x

    df['context'] = df['context'].parallel_apply(truncate_context)

    def reddit_clean(x):
        return p.tokenize(redditcleaner.clean(x))

    def reddit_batch_clean(x):
        return [p.tokenize(redditcleaner.clean(e)) for e in x]

    df['final_comment'] = df['final_comment'].parallel_apply(reddit_clean)
    df['context'] = df['context'].parallel_apply(reddit_batch_clean)
    df['bool_derail'] = df['bool_derail'].astype(int)

    from openprompt.data_utils import InputExample
    def create_input_example(row):
        meta = {
            'subreddit': row['subreddit'],
            'rule': row['rule_texts'],
            'cat': row['cats'],
        }
        for i in range(max_context_size):
            meta[f'comment{i}'] = row['context'][i]
        meta[f'comment{max_context_size}'] = row['final_comment']
        return InputExample(meta=meta, label=row['bool_derail'])

    data = df.parallel_apply(create_input_example, axis=1).tolist()
    for i, each in enumerate(data):
        each.guid = i

    features = df['cats']

    return data, features


def data_loader(phase,
                batch_size,
                n_few_shot=0,
                cat='all',
                comm='all',
                max_context_size=5,
                max_n_tokens=128,
                n_workers=4,
                seed=2022
                ):
    shuffle = True if phase == 'train' else False
    dataset = NormVioSeq(phase=phase,
                         n_few_shot=n_few_shot,
                         cat=cat,
                         comm=comm,
                         max_context_size=max_context_size,
                         max_n_tokens=max_n_tokens,
                         n_workers=n_workers,
                         seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    return loader


if __name__ == '__main__':
    create_normvio_prompt_dataset('test')
