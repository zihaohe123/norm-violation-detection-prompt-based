import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='all_cat_all_comm', help='description of current training')

    # data config
    parser.add_argument('--max_context_size', type=int, default=5, help='max number of preceding comments')
    parser.add_argument('--max_n_tokens', type=int, default=384, help='max number of tokens in the template')
    parser.add_argument('--src_cat', type=str, default='all',
                        choices=('all', 'incivility', 'harassment', 'spam', 'format', 'content',
                                 'off-topic', 'hatespeech', 'trolling', 'meta-rules',
                                 '~incivility', '~harassment', '~spam', '~format', '~content',
                                 '~off-topic', '~hatespeech', '~trolling', '~meta-rules',
                                 ),
                        help='source category. ~category indicates all but this category.')
    parser.add_argument('--tgt_cat', type=str, default='all',
                        choices=('all', 'incivility', 'harassment', 'spam', 'format', 'content',
                                 'off-topic', 'hatespeech', 'trolling', 'meta-rules'),
                        help='target category. ~category indicates all but this category.')
    parser.add_argument('--src_comm', type=str, default='all',
                        choices=('all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                                 'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm',
                                 '~Coronavirus', '~CanadaPolitics', '~LabourUK', '~TexasPolitics',
                                 '~classicwow', '~Games', '~RPClipsGTA', '~heroesofthestorm'
                                 ),
                        help='source community. ~community indicates all but this category.'
                        )
    parser.add_argument('--tgt_comm', type=str, default='all',
                        choices=('all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                                 'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm'),
                        help='target community. ~community indicates all but this category.'
                        )
    parser.add_argument('--n_few_shot', type=int, default=0,
                        help='number of training examples in the few shot setting. '
                             '0 indicates not doing few-shot and using all examples instead.')

    # training config
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    print(args)

    prefix = f'results/{args.description}/seed={args.seed}'
    os.makedirs(prefix, exist_ok=True)

    args_dict = args.__dict__
    with open(f'{prefix}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(json.dumps(args_dict, indent=2))

    from trainer_prompt import Trainer
    trainer = Trainer(args)
    trainer.train()
