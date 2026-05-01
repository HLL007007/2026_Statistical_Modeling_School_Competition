# ==========================================
# 1. Imports - 导入必要的库
# ==========================================
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

# 机器学习模型与工具
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, balanced_accuracy_score

# 忽略警告信息，保持输出界面整洁
warnings.filterwarnings('ignore')

# ==========================================
# 2. Helper Functions - 全局设置与辅助函数
# ==========================================
# 设置随机种子，保证每次运行的结果一致（比赛可复现性极其重要！）
SEED = 43
random.seed(SEED)
np.random.seed(SEED)


# Matplotlib 全局参数设置
plt.style.use('bmh') # 截图中的 bmh 风格
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['font.size'] = 18

# Pandas 全局显示设置 (显示所有列和行，不换行)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('display.expand_frame_repr', False)

sns.set()

# --- 辅助函数 1：BMI 分类函数 ---
# 根据图1中的说明：0: Underweight, 1: Normal, 2: Overweight, 3: Obesity 1, 4: Obesity 2, 5: Morbid obesity
def cat_bmi(bmi):
    if bmi < 18.5:
        return int(0) # 0: underweight
    elif 18.5 <= bmi < 25:
        return int(1) # 1: normal
    elif 25 <= bmi < 30:
        return int(2) # 2: overweight (对应截图的 elevated/overweight)
    elif 30 <= bmi < 35:
        return int(3) # 3: obesity 1
    elif 35 <= bmi < 40:
        return int(4) # 4: obesity 2
    else: # bmi >= 40
        return int(5) # 5: morbid obesity

# --- 辅助函数 2：柱状图数值自动标注函数 ---
def annot_plot(plot):
    for p in plot.patches:
        plot.annotate(format(int(p.get_height())),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha = 'center', va = 'center',
                      xytext = (0, 9),
                      textcoords = 'offset points',
                      fontsize = 14)
    return None


# ==========================================
# 3. Load Data - 加载数据集
# ==========================================

try:
    df1 = pd.read_csv('data\\cardio_train.csv', sep=';', index_col='id')
    print("数据加载成功！")
    print(f"数据集维度 (行, 列): {df1.shape}")
except FileNotFoundError:
    print("错误：未找到数据文件，请检查文件路径是否正确！")

# 预览前5行数据
print(df1.head())