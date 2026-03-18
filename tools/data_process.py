import json

def remove_ear_annotations(input_file, output_file):
    """
    删除posetrack_train.json中的左耳和右耳标注数据
    """
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 更新categories中的keypoints列表，移除左耳和右耳
    if 'categories' in data:
        for category in data['categories']:
            if 'keypoints' in category:
                # 移除左耳和右耳
                category['keypoints'] = [kp for kp in category['keypoints'] if kp not in ['left_ear', 'right_ear']]
    
    # 2. 处理annotations中的keypoints数据
    if 'annotations' in data:
        for annotation in data['annotations']:
            if 'keypoints' in annotation:
                keypoints = annotation['keypoints']
                # 计算新的keypoints数组（移除左耳和右耳的数据）
                new_keypoints = []
                for i in range(len(keypoints) // 3):
                    # 跳过左耳（索引3）和右耳（索引4）
                    if i != 3 and i != 4:
                        start_idx = i * 3
                        new_keypoints.extend(keypoints[start_idx:start_idx+3])
                annotation['keypoints'] = new_keypoints
    
    # 保存为新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成，已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "posetrack_train.json"
    output_file = "posetrack_train_fixed.json"
    remove_ear_annotations(input_file, output_file)