import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import time
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成模拟数据
def generate_sample_data(n_samples=100, grid_size=16, n_clusters=5):
    """生成100个16×16的模拟数据，每个数据点属于5个不同的簇"""
    np.random.seed(42)
    data = np.zeros((n_samples, grid_size * grid_size))
    
    # 为每个簇定义不同的分布特征
    cluster_centers = [
        np.random.normal(loc=0.2, scale=0.1, size=grid_size*grid_size),  # 簇1：低强度中心
        np.random.normal(loc=0.5, scale=0.1, size=grid_size*grid_size),  # 簇2：中等强度中心
        np.random.normal(loc=0.8, scale=0.1, size=grid_size*grid_size),  # 簇3：高强度中心
        np.random.normal(loc=1.5, scale=0.1, size=grid_size*grid_size),  # 簇4：双极分布
        np.random.normal(loc=3.2, scale=0.1, size=grid_size*grid_size)   # 簇5：反向双极分布
    ]
    
    # 为每个样本分配到一个簇，并添加随机噪声
    for i in range(n_samples):
        cluster_idx = i % n_clusters
        data[i] = cluster_centers[cluster_idx] + np.random.normal(0, 0.1, grid_size*grid_size)
    
    # 确保所有值在[0,1]范围内
    data = np.clip(data, 0, 1)
    
    return data, grid_size

# 聚类分析
def perform_clustering(data, n_clusters=5, eps=0.5, min_samples=5):
    """执行多种聚类算法并返回结果"""
    results = {}
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 定义要比较的聚类方法
    clustering_methods = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
        "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
        "SpectralClustering": SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    }
    
    # 创建保存结果的目录
    if not os.path.exists("clustering_results"):
        os.makedirs("clustering_results")
    
    # 对每种聚类方法进行训练和评估
    for name, model in clustering_methods.items():
        print(f"\n=== 正在使用 {name} 进行聚类 ===")
        start_time = time.time()
        
        # 执行聚类
        if name == "DBSCAN":
            # DBSCAN可能生成不同数量的簇，需要特殊处理
            labels = model.fit_predict(scaled_data)
            unique_labels = np.unique(labels)
            actual_n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            
            if actual_n_clusters != n_clusters:
                print(f"警告: DBSCAN生成了{actual_n_clusters}个簇，而不是{n_clusters}个")
        elif name == "GMM":
            model.fit(scaled_data)
            labels = model.predict(scaled_data)
        else:
            labels = model.fit_predict(scaled_data)
        
        # 计算轮廓系数（如果有多个簇）
        if len(np.unique(labels)) > 1:
            try:
                score = silhouette_score(scaled_data, labels)
                print(f"轮廓系数: {score:.4f}")
            except:
                print("无法计算轮廓系数")
                score = np.nan
        else:
            print("只有一个簇，无法计算轮廓系数")
            score = np.nan
        
        # 保存结果
        results[name] = {
            "labels": labels,
            "silhouette_score": score,
            "time": time.time() - start_time
        }
        
        print(f"{name} 聚类完成，耗时: {time.time() - start_time:.2f}秒")
    
    return results

# 可视化聚类结果
def visualize_clustering_results(data, results, grid_size, n_clusters=5):
    """可视化每个样本的聚类结果"""
    n_samples = data.shape[0]
    n_methods = len(results)
    
    # 为每个簇定义颜色
    colors = plt.cm.get_cmap('viridis', n_clusters)
    
    # 选择一些样本进行可视化（例如前9个）
    samples_to_visualize = min(9, n_samples)
    
    # 创建一个大的图形
    fig, axes = plt.subplots(samples_to_visualize, n_methods + 1, figsize=(5 * (n_methods + 1), 5 * samples_to_visualize))
    
    for i in range(samples_to_visualize):
        # 显示原始数据
        ax = axes[i, 0]
        img = data[i].reshape(grid_size, grid_size)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'样本 {i+1} (原始数据)')
        ax.axis('off')
        
        # 为每个聚类方法显示结果
        method_idx = 1
        for method_name, result in results.items():
            ax = axes[i, method_idx]
            
            # 获取该样本的聚类标签
            label = result['labels'][i]
            
            # 创建一个16×16的网格，其中每个元素的颜色由其聚类标签决定
            # 这里假设每个16×16的元素属于同一个簇
            cluster_grid = np.full((grid_size, grid_size), label)
            
            # 显示聚类结果
            im = ax.imshow(cluster_grid, cmap=colors, vmin=0, vmax=n_clusters-1)
            ax.set_title(f'{method_name}: 簇 {label}')
            ax.axis('off')
            
            method_idx += 1
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(n_clusters))
    cbar.set_label('簇标签')
    
    plt.tight_layout()
    plt.savefig('clustering_results/combined_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# 可视化每个聚类算法的整体性能
def visualize_performance(results):
    """可视化每种聚类算法的性能指标"""
    methods = list(results.keys())
    silhouette_scores = [results[m]['silhouette_score'] for m in methods]
    times = [results[m]['time'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 轮廓系数比较
    ax1.bar(methods, silhouette_scores, color='skyblue')
    ax1.set_title('轮廓系数比较')
    ax1.set_ylabel('轮廓系数')
    ax1.set_ylim(0, 1)  # 轮廓系数范围是[-1,1]，但通常好的聚类在0-1之间
    
    # 添加数值标签
    for i, v in enumerate(silhouette_scores):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 运行时间比较
    ax2.bar(methods, times, color='lightgreen')
    ax2.set_title('运行时间比较')
    ax2.set_ylabel('时间 (秒)')
    
    # 添加数值标签
    for i, v in enumerate(times):
        ax2.text(i, v + 0.01, f'{v:.3f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('clustering_results/performance_comparison.png', dpi=300)
    plt.show()

# 主函数
def main():
    print("=== 开始聚类分析 ===")
    
    # 生成模拟数据
    data, grid_size = generate_sample_data()
    print(f"生成数据: {data.shape} - {data.shape[0]}个样本，每个样本{grid_size}×{grid_size}")
    
    # 执行聚类
    n_clusters = 5  # 聚类数量
    eps = 0.8       # DBSCAN的eps参数
    min_samples = 5 # DBSCAN的min_samples参数
    
    results = perform_clustering(data, n_clusters, eps, min_samples)
    
    # 可视化聚类结果
    visualize_clustering_results(data, results, grid_size, n_clusters)
    
    # 可视化性能比较
    visualize_performance(results)
    
    print("\n=== 聚类分析完成 ===")
    print("结果已保存至 clustering_results 目录")

if __name__ == "__main__":
    main()