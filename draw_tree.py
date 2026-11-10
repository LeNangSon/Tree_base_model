import matplotlib.pyplot as plt 
from DecisionTree import Decision_Tree 
import kaggle 
import pandas as pd 
import os 
from sklearn.preprocessing import LabelEncoder
from graphviz import Digraph

kaggle.api.authenticate() 

#dataset_name = 'fredericobreno/play-tennis'
#kaggle.api.dataset_download_files(dataset_name , path='/home/amiyasociu/HocTap/Tree_base_model' , unzip=True) 
#print("ok") 
#DATA
data_dir = '/home/amiyasociu/HocTap/Tree_base_model'
csv_path = os.path.join(data_dir , 'play_tennis.csv') 
df = pd.read_csv(csv_path)

#drop days 
cols_to_drop = [c for c in df.columns if c.lower() in ['day']]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)


#trace it into number
encoders = {}
for col in df.columns:
    le = LabelEncoder() 
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print(df.head())

target_col = df.columns[-1]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 
feature_names = list(df.columns[:-1])
class_names = [str(c) for c in encoders[target_col].classes_] 

clf = Decision_Tree(    
    min_sample_split=2, 
    max_depth=3, 
    min_impurity_decrease=0.0)
clf.fit(X , y) #train model 
y_pred = clf.predict(X)
acc = (y_pred == y).mean() * 100
print(acc) 


#draw_tree (test gemini-flash-3.8) 
dot = Digraph(comment='Decision Tree', format='png')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E8F5E9', fontname='Arial')

# Hàm đệ quy siêu ngắn: chỉ cần chỉ ra quan hệ Cha -> Con
def build_graph(node):
    if node is None:
        return
    
    node_id = str(id(node))
    
    # 1. Nếu là Node lá
    if node.value is not None:
        label = f"Play: {class_names[node.value]}"
        dot.node(node_id, label, fillcolor='#C8E6C9') # Màu xanh lá đậm hơn cho node kết quả
    # 2. Nếu là Node rẽ nhánh
    else:
        label = f"{feature_names[node.feature_index]} <= {node.threshold:.2f}"
        dot.node(node_id, label)

    # Nối dây sang nhánh trái (True)
    if node.left:
        dot.edge(node_id, str(id(node.left)), label=" True", color="green")
        build_graph(node.left)

    # Nối dây sang nhánh phải (False)
    if node.right:
        dot.edge(node_id, str(id(node.right)), label=" False", color="red")
        build_graph(node.right)

# Gọi vẽ cây từ node gốc của bạn
build_graph(clf.root)

# Xuất thẳng ra file ảnh PNG đẹp chuẩn chỉnh
dot.render('/home/amiyasociu/HocTap/Tree_base_model/tree_graphviz', cleanup=True)
print("-> Đã xuất ảnh cây tại: /home/amiyasociu/HocTap/Tree_base_model/tree_graphviz.png")