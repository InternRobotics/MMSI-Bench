from vlmeval.smp import *
from vlmeval.dataset import SUPPORTED_DATASETS

# 存储数据集各个类别的样本数
category_counts = {}

def get_score(model, dataset):
    global category_counts
    
    file_name = f'outputs/{model}/{model}_{dataset}'
    if listinstr([
        'CCBench', 'MMBench', 'SEEDBench_IMG', 'MMMU', 'ScienceQA', 
        'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK', 'VisOnlyQA-VLMEvalKit'
    ], dataset):
        file_name += '_acc.csv'
    elif dataset == 'MMSI_Bench_Circular':
        file_name += '_combined_acc.csv'  # 使用新的 combined_acc.csv 文件
    elif listinstr(['MME', 'Hallusion', 'LLaVABench'], dataset):
        file_name += '_score.csv'
    elif listinstr(['MMVet', 'MathVista'], dataset):
        file_name += '_gpt-4-turbo_score.csv'
    elif listinstr(['COCO', 'OCRBench'], dataset):
        file_name += '_score.json'
    elif listinstr(['Spatial457'], dataset):
        file_name += '_score.json'
    elif listinstr(['MMSI_Bench'], dataset):
        file_name += '_score.xlsx'
    else:
        raise NotImplementedError
    if not osp.exists(file_name):
        print(f"文件未找到: {file_name}")
        return {}
    
    data = load(file_name)
    ret = {}
    if dataset == 'CCBench':
        ret[dataset] = data['Overall'][0] * 100
    elif dataset == 'MMBench':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_EN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_EN'] = a * 100
    elif dataset == 'MMBench_CN':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_CN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_CN'] = a * 100
    elif listinstr(['SEEDBench', 'ScienceQA', 'MMBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK'], dataset):
        ret[dataset] = data['Overall'][0] * 100
    elif 'MME' == dataset:
        ret[dataset] = data['perception'][0] + data['reasoning'][0]
    elif 'MMVet' == dataset:
        data = data[data['Category'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'HallusionBench' == dataset:
        data = data[data['split'] == 'Overall']
        for met in ['aAcc', 'qAcc', 'fAcc']:
            ret[dataset + f' ({met})'] = float(data.iloc[0][met])
    elif 'MMMU' in dataset:
        data = data[data['split'] == 'validation']
        ret['MMMU (val)'] = float(data.iloc[0]['Overall']) * 100
    elif 'MathVista' in dataset:
        data = data[data['Task&Skill'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'LLaVABench' in dataset:
        data = data[data['split'] == 'overall'].iloc[0]
        ret[dataset] = float(data['Relative Score (main)'])
    elif 'OCRBench' in dataset:
        ret[dataset] = data['Final Score']
    elif dataset == "VisOnlyQA-VLMEvalKit":
        for n, a in zip(data['split'], data['Overall']):
            ret[f'VisOnlyQA-VLMEvalKit_{n}'] = a * 100
    elif 'Spatial457' in dataset:
        ret["All"] = data["score"] * 100
        for level in ["L1_single", "L2_objects", "L3_2d_spatial", "L4_occ",
                        "L4_pose", "L5_6d_spatial", "L5_collision"]:
            ret[f"{dataset} - {level}"] = data[f"{level}_score"] * 100
    elif dataset == 'MMSI_Bench_Circular':
        # 简化处理 - 直接读取 combined_acc.csv 文件
        data = pd.read_csv(file_name, index_col=0)
        
        # 获取总体结果
        ret['MMSI_Bench (Vanilla)'] = data.loc['Overall', 'Vanilla'] * 100
        ret['MMSI_Bench (Circular)'] = data.loc['Overall', 'Circular'] * 100
        
        # 获取类别结果 - 先添加Vanilla结果，然后添加Circular结果
        for idx in data.index:
            if idx == 'Overall':
                continue
            category = idx
            ret[f'MMSI_Bench (Vanilla) - {category}'] = data.loc[category, 'Vanilla'] * 100
        
        for idx in data.index:
            if idx == 'Overall':
                continue
            category = idx
            ret[f'MMSI_Bench (Circular) - {category}'] = data.loc[category, 'Circular'] * 100
    elif dataset == 'MMSI_Bench':
        # Calculate overall accuracy from the score column (0 or 1 for each question)
        if 'score' in data.columns:
            # Overall accuracy is the mean of all scores
            overall_acc = data['score'].mean() * 100
            ret[dataset] = overall_acc
            
            # 计算每个类别的样本数量
            if 'category' in data.columns and dataset not in category_counts:
                category_counts[dataset] = data['category'].value_counts().to_dict()
            
            # Calculate category-wise accuracies
            if 'category' in data.columns:
                category_results = data.groupby('category')['score'].mean() * 100
                for cat, score in category_results.items():
                    if not pd.isna(cat) and cat != 'nan':
                        ret[f'{dataset} - {cat}'] = score
    return ret

def get_category_counts(model, dataset):
    """
    从原始数据文件中获取各个类别的样本数量
    """
    if dataset != 'MMSI_Bench_Circular':
        return {}
    
    # 尝试从原始文件获取类别样本数量
    counts = {}
    try:
        # 尝试读取 vanilla_result.xlsx 文件，这个文件应该包含所有原始样本
        file_path = f'outputs/{model}/{model}_{dataset}_vanilla_result.xlsx'
        if osp.exists(file_path):
            data = load(file_path)
            if 'category' in data.columns:
                counts = data['category'].value_counts().to_dict()
                logging.info(f"成功从 {file_path} 获取类别样本数量")
        else:
            # 尝试读取原始评估文件
            file_path = f'outputs/{model}/{model}_{dataset}.xlsx'
            if osp.exists(file_path):
                data = load(file_path)
                # 对于 MMSI_Bench_Circular，需要先处理 g_index
                if 'g_index' in data.columns and 'category' in data.columns:
                    # 按 g_index 分组，只统计每个组中的第一个样本
                    unique_samples = data.drop_duplicates(subset=['g_index'])
                    counts = unique_samples['category'].value_counts().to_dict()
                    logging.info(f"成功从 {file_path} 获取类别样本数量")
    except Exception as e:
        logging.warning(f"获取类别样本数量失败: {e}")
    
    return counts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', default=[])
    parser.add_argument("--model", type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args

def gen_table(models, datasets):
    res = defaultdict(dict)
    for m in models:
        for d in datasets:
            try:
                res[m].update(get_score(m, d))
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                logging.warning(f'Missing Results for Model {m} x Dataset {d}')
    
    # 获取所有键并排序
    keys = []
    for m in models:
        for d in res[m]:
            keys.append(d)
    keys = list(set(keys))
    
    # 对keys进行排序，使得相关条目分组显示：先显示总体结果，然后是Vanilla细分结果，最后是Circular细分结果
    def sort_key(item):
        if "MMSI_Bench (Vanilla)" == item:
            return (0, 0)  # 总体Vanilla结果排在最前面
        elif "MMSI_Bench (Circular)" == item:
            return (0, 1)  # 总体Circular结果排第二
        elif "MMSI_Bench (Vanilla) -" in item:
            return (1, item)  # Vanilla细分结果排在中间
        elif "MMSI_Bench (Circular) -" in item:
            return (2, item)  # Circular细分结果排在最后
        else:
            return (3, item)  # 其他结果
    
    keys.sort(key=sort_key)
    
    # 创建最终数据
    final = defaultdict(list)
    for m in models:
        final['Model'].append(m)
        for k in keys:
            if k in res[m]:
                final[k].append(res[m][k])
            else:
                final[k].append(None)
    final = pd.DataFrame(final)
    final = final.set_index('Model').T.reset_index().rename(columns={'index': 'DataSet/Category'})
    
    # 显示类别样本数量信息
    if category_counts:
        print("\n=== 数据集类别样本统计 ===")
        for dataset, counts in category_counts.items():
            if not counts:  # 跳过空的类别统计
                continue
                
            total_samples = sum(counts.values())
            print(f"\n{dataset} 总样本数: {total_samples}")
            category_info = []
            for cat, count in sorted(counts.items()):
                if not pd.isna(cat) and cat != 'nan':
                    category_info.append(f"{cat}: {count}题")
            
            # 每行打印3个类别信息
            for i in range(0, len(category_info), 3):
                print("  ".join(category_info[i:i+3]))
    
    # 简单保存结果
    dump(final, 'summ.csv')
    
    # 使用 tabulate 显示，确保表格格式整齐
    print(tabulate(final, headers='keys', tablefmt='psql', floatfmt='.2f', showindex=False))

if __name__ == '__main__':
    args = parse_args()
    if args.data == []:
        args.data = list(SUPPORTED_DATASETS)
    gen_table(args.model, args.data)