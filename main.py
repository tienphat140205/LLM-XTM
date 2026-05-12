import os
import yaml
import scipy.io
from scipy.io import loadmat
from runners.Runner import Runner
import argparse

from utils.data import file_utils
from utils.data.TextData import DatasetHandler
from utils import miscellaneous, seed
# from CNPMI.CNPMI import calcwcngram_complete, calcwcngram, calc_assoc
from CNPMI.CNPMI import *
from utils.TU import *
from utils.eval import *
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_topic', type=int, default=50)

    parser.add_argument('--ref_corpus_config', type=str, default="CNPMI/configs/ref_corpus/en_zh.yaml")
    parser.add_argument('--metric', type=str, default='npmi')

    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use')
    parser.add_argument('--warmStep', default=0, type=int)
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--gemini_api_key', type=str, default=None,
                        help='LLM API key for cross-lingual topic refinement')
    parser.add_argument('--llm_provider', type=str, default=None, choices=['openai', 'gemini'],
                        help='LLM provider for topic refinement: openai means OpenAI-compatible API')
    parser.add_argument('--llm_model', type=str, default=None,
                        help='LLM model for topic refinement')
    parser.add_argument('--llm_base_url', type=str, default=None,
                        help='Base URL for OpenAI-compatible topic refinement')
    parser.add_argument('--refinement_rounds', type=int, default=5,
                        help='Number of self-consistent refinement rounds (R)')
    parser.add_argument('--refine_frequency', type=int, default=10,
                        help='Frequency of refinement during training (every N epochs after warmStep)')
    parser.add_argument('--refine_weight', type=float, default=20000,
                        help='Weight for refinement loss (0 disables refinement loss)')
    parser.add_argument('--topic_sim_weight', type=float, default=100,
                        help='Weight for topic embedding similarity loss (0 disables topic similarity loss)')



    # Add missing arguments used in the code
    parser.add_argument('--wandb_prj', type=str, default='ARR-October', help='Wandb project name')

    args = parser.parse_args()
    
    if args.llm_provider is None:
        args.llm_provider = os.getenv('LLM_PROVIDER', 'openai')

    # Backward-compatible name: gemini_api_key is used as the refinement API key.
    if args.gemini_api_key is None:
        if args.llm_provider == 'gemini':
            args.gemini_api_key = os.getenv('GEMINI_API_KEY')
        else:
            args.gemini_api_key = os.getenv('LLM_API_KEY') or os.getenv('NVIDIA_API_KEY') or os.getenv('GEMINI_API_KEY')

    if args.llm_model is None:
        if args.llm_provider == 'gemini':
            args.llm_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        else:
            args.llm_model = os.getenv('LLM_MODEL', 'qwen/qwen3-coder-480b-a35b-instruct')

    if args.llm_base_url is None:
        args.llm_base_url = os.getenv('LLM_BASE_URL', 'https://integrate.api.nvidia.com/v1')

    return args


def export_beta(beta, vocab, output_prefix, lang):
    num_top_word = 15  # Standard evaluation uses 15 words
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}/T{num_top_word}_{lang}.txt')
    return topic_str_list

RESULT_DIR= 'output'
def main():
    args = parse_args()

    args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')
    args.warmStep = args.epochs - args.llm_step
    args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')

    if args.lang2 == "ja":
        args.ref_corpus_config = "CNPMI/configs/ref_corpus/en_ja.yaml"
        print(f"Setting reference corpus config to {args.ref_corpus_config} for Japanese")
    
    prj = args.wandb_prj 
    current_time = miscellaneous.get_current_datetime()
    # Create output path based on model type
    if args.model == 'InfoCTM':
        model_params = f"weightMI_{args.weight_MI}"
    elif args.model == 'NMTM':
        model_params = f"lam_{args.lam}"
    elif args.model == 'XTRA':
        # Summarize key hyperparameters for experiment grouping
        model_params = f"K{args.num_topic}_sd{getattr(args, 'share_dim', 1000)}_wc{getattr(args, 'weight_cluster', 0.0)}_wb{getattr(args, 'weight_beta', 0.0)}_win{getattr(args, 'weight_InfoNCE', 0.0)}"
    else:
        # Fallback for other models
        model_params = "default"
    
    output_prefix = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" + str(args.dataset), 
                    model_params, current_time)
    miscellaneous.create_folder_if_not_exist(output_prefix)
    seed.seedEverything(args.seed)
    
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    wandb.init(project=prj, config=args)
    wandb.log({'time_stamp': current_time})

    print('\n' + yaml.dump(vars(args), default_flow_style=False))
    

    dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2, args.num_topic, device=args.device, model_name=args.model)

    args.doc_embeddings_en = getattr(dataset_handler, 'doc_embeddings_en', None)
    args.doc_embeddings_cn = getattr(dataset_handler, 'doc_embeddings_cn', None)


    args.vocab_size_en = len(dataset_handler.vocab_en)
    args.vocab_size_cn = len(dataset_handler.vocab_cn)


    args.vocab_en = dataset_handler.vocab_en
    args.vocab_cn = dataset_handler.vocab_cn

    # Pass word embeddings to args (optional for XTRA)
    args.word_embeddings_en = getattr(dataset_handler, 'word_embeddings_en', None)
    args.word_embeddings_cn = getattr(dataset_handler, 'word_embeddings_cn', None)

    # Pass only required attributes per model
    if args.model == 'InfoCTM':
        args.trans_matrix_en = getattr(dataset_handler, 'trans_matrix_en', None)
        args.pretrained_WE_en = getattr(dataset_handler, 'pretrained_WE_en', None)
        args.pretrained_WE_cn = getattr(dataset_handler, 'pretrained_WE_cn', None)
    elif args.model == 'NMTM':
        args.Map_en2cn = getattr(dataset_handler, 'Map_en2cn', None)
        args.Map_cn2en = getattr(dataset_handler, 'Map_cn2en', None)

    # XTRA initialization support
    if args.model == 'XTRA':
        args.beta_en = getattr(dataset_handler, 'beta_en', None)
        args.beta_cn = getattr(dataset_handler, 'beta_cn', None)
        args.mu_prior = getattr(dataset_handler, 'mu_prior', None)
        args.var_prior = getattr(dataset_handler, 'var_prior', None)
    

    runner = Runner(args)

    # Train with refinement if API key available
    beta_en, beta_cn = runner.train(dataset_handler.train_loader)

    # Get refined topics if available
    refined_topics = getattr(runner, 'refined_topics', None)
    high_confidence_topics = getattr(runner, 'high_confidence_topics', None)

    # Always evaluate using original beta top words (never refined)
    topic_str_list_en = export_beta(beta_en, dataset_handler.vocab_en, output_prefix, lang=args.lang1)
    topic_str_list_cn = export_beta(beta_cn, dataset_handler.vocab_cn, output_prefix, lang=args.lang2)
    print(f"Using original beta for evaluation ({len(topic_str_list_en)} topics)")

    for i in range(len(topic_str_list_en)):
        print(topic_str_list_en[i])
        print(topic_str_list_cn[i])

    train_theta_en, train_theta_cn = runner.test(dataset_handler.train_loader.dataset)
    test_theta_en, test_theta_cn = runner.test(dataset_handler.test_loader.dataset)

    rst_dict = {
        'beta_en': beta_en,
        'beta_cn': beta_cn,
        'train_theta_en': train_theta_en,
        'train_theta_cn': train_theta_cn,
        'test_theta_en': test_theta_en,
        'test_theta_cn': test_theta_cn,
    }

    scipy.io.savemat(f'{output_prefix}/rst.mat', rst_dict)
    
    # Calculate CNPMI
    parallel_corpus_tuples = file_utils.read_yaml(args.ref_corpus_config)['parallel_corpus_tuples']
    num_top_word = 15  # Standard NPMI calculation uses 15 words

    sep_token = '|'

    topics1 = read_texts_cnpmi(f'{output_prefix}/T{num_top_word}_{args.lang1}.txt')
    topics1 = split_text_word_cnpmi(topics1)
    topics2 = read_texts_cnpmi(f'{output_prefix}/T{num_top_word}_{args.lang2}.txt')
    topics2 = split_text_word_cnpmi(topics2)

    num_topic = len(topics1)
    num_top_word = len(topics1[0])

    vocab1 = set([])
    vocab2 = set([])
    word_pair_list = list()
    for k in range(num_topic):
        for i in range(num_top_word):
            w1 = topics1[k][i]
            vocab1.add(w1)
            for j in range(num_top_word):
                w2 = topics2[k][j]
                vocab2.add(w2)
                word_pair_list.append(f'{w1}{sep_token}{w2}')

    word_pair_list = tuple(word_pair_list)
    vocab1 = sorted(list(vocab1))
    vocab2 = sorted(list(vocab2))

    pool = Pool()
    for i, cp in enumerate(parallel_corpus_tuples):
        if not os.path.exists(cp[0]):
            raise FileNotFoundError(cp[0])
        if not os.path.exists(cp[1]):
            raise FileNotFoundError(cp[1])

        param_list = (cp, vocab1, vocab2, word_pair_list, sep_token)
        pool.apply_async(calcwcngram, param_list, callback=calcwcngram_complete)

    # wait for the subprocesses.
    pool.close()
    pool.join()

    topic_assoc = list()
    window_total = float(global_word_count[WTOTALKEY])
    for word_pair in word_pair_list:
        topic_assoc.append(calc_assoc(word_pair, window_total, sep_token, metric=args.metric))

    result = float(sum(topic_assoc)) / len(topic_assoc)
    print(f"CNPMI: {result:.5f}")
    wandb.log({"CNPMI": result})
    
    #Calculate TU
    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang1}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    tu_lang1 = TU_eva(texts)
    print(f"TU_{args.lang1}: {tu_lang1:.5f}")
    wandb.log({f"TU_{args.lang1}": tu_lang1})

    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang2}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    tu_lang2 = TU_eva(texts)
    print(f"TU_{args.lang2}: {tu_lang2:.5f}")
    wandb.log({f"TU_{args.lang2}": tu_lang2})

    tu_average = (tu_lang1 + tu_lang2) / 2
    print(f"TU_Average: {tu_average:.5f}")
    wandb.log({"TU_Average": tu_average})
    #----------Eval theta and more--------------
    dataset_name = args.dataset # Example: Change to your dataset
    model_name = args.model      # Example: Change to your model name
    num_topics = args.num_topic             # Example: Number of topics used in the model
    # num_top_words_display = 15   # Number of top words in the output files (T15)

    # Construct paths
    # base_output_dir = f"./output/{dataset_name}"
    base_data_dir = f"./data/{dataset_name}"
    # mat_path = f"{base_output_dir}/{model_name}_K{num_topics}_rst.mat"
    mat_path = f'{output_prefix}/rst.mat'

    # Paths for text data and labels
    en_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang1}.txt'
    cn_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang2}.txt'

    en_corpus_path = f"{base_data_dir}/train_texts_en.txt" 
    train_labels_en_path = f"{base_data_dir}/train_labels_en.txt"
    train_labels_cn_path = f"{base_data_dir}/train_labels_cn.txt"
    test_labels_en_path = f"{base_data_dir}/test_labels_en.txt"
    test_labels_cn_path = f"{base_data_dir}/test_labels_cn.txt"
    if args.lang2 == "ja":
        train_labels_cn_path = f"{base_data_dir}/train_labels_ja.txt"
        test_labels_cn_path = f"{base_data_dir}/test_labels_ja.txt"

    print(f"--- Evaluating Model: {model_name}, Dataset: {dataset_name}, K={num_topics} ---")

    print("\n--- Loading Data ---")
    train_labels_en = load_labels_txt(train_labels_en_path)
    train_labels_cn = load_labels_txt(train_labels_cn_path)
    test_labels_en = load_labels_txt(test_labels_en_path)
    test_labels_cn = load_labels_txt(test_labels_cn_path)
    if any(arr.size == 0 for arr in [train_labels_en, train_labels_cn, test_labels_en, test_labels_cn]):
        print("Error: Failed to load one or more label files. Exiting.")
        exit()
    print("Labels loaded successfully.")

    # Load results matrix (.mat file)
    try:
        mat = loadmat(mat_path)
        train_theta_en = mat["train_theta_en"]
        train_theta_cn = mat["train_theta_cn"]
        test_theta_en = mat["test_theta_en"]    
        test_theta_cn = mat["test_theta_cn"]
        print(f"Results matrix loaded successfully from {mat_path}.")
    except FileNotFoundError:
        print(f"Error: Results matrix file not found at {mat_path}. Exiting.")
        exit()
    except KeyError as e:
        print(f"Error: Key {e} not found in results matrix {mat_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the .mat file: {e}. Exiting.")
        exit()

    print("Loading text data for Coherence/Diversity...")
    en_top_words_list = split_text_word(en_top_words_path)
    cn_top_words_list = split_text_word(cn_top_words_path)



    print("\n================= Classification (Real Theta) =================")
    cls_results = crosslingual_cls(
        train_theta_en, train_theta_cn,
        test_theta_en, test_theta_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    print_results(cls_results)
    wandb.log({"intra_en": cls_results["intra_en"]})
    wandb.log({"intra_cn": cls_results["intra_cn"]})
    wandb.log({"cross_en": cls_results["cross_en"]})
    wandb.log({"cross_cn": cls_results["cross_cn"]})
    
if __name__ == '__main__':
    main()
