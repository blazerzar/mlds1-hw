from collections import Counter
from dataclasses import dataclass
from json import load

import matplotlib.pyplot as plt
import numpy as np
import vispy  # type: ignore
import vispy.io as io  # type: ignore
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from vispy import scene  # type: ignore
from vispy.scene import visuals  # type: ignore

LINE_WIDTH = 3.220
SHOW_PLOTS = True
SAVE_PLOTS = False


@dataclass
class Article:
    title: str
    keywords: list[str]


def load_data(path: str) -> list[Article]:
    """Load data from a JSON file and return a list of Article objects."""
    data = load(open(path, 'r'))
    articles = [Article(a['title'], a['gpt_keywords']) for a in data]
    return articles


def preprocess_keywords(keywords: list[str]) -> list[str]:
    to_remove = '.,:;!?()[]{}-"\'' + '0123456789'
    keywords = [''.join(c for c in k if not c in to_remove) for k in keywords]
    keywords = [k.strip().replace(' ', '_') for k in keywords if len(k) > 2]
    return keywords


def create_features(articles: list[Article]) -> tuple[NDArray, list[str]]:
    """Create a matrix of features from a list of Article objects. We use
    tf-idf of keywords that appear in at least 20 articles."""
    corpus = [' '.join(a.keywords) for a in articles]
    vectorizer = TfidfVectorizer(min_df=20)
    X = vectorizer.fit_transform(corpus)

    return X.toarray(), vectorizer.get_feature_names_out()


def pca_task(X: NDArray, keyword_names: list[str]) -> None:
    """Transform the data using PCA with the first 3 components. Plot the
    transformed data points and overlay a loading plot on it."""
    pca = PCA(n_components=3, random_state=1)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    scatter.set_data(
        X_pca, edge_color=(0.1, 0.1, 0.1, 0.4), face_color=(0.9, 0.5, 0.1, 0.2), size=6
    )
    view.add(scatter)
    view.camera = 'turntable'

    # Find keywords aligned with the following vectors
    vectors = np.array(
        [
            [-5, 15, -10],
            [5, -8, -17],
            [15, 15, 30],
            [-3, -3, 3],
            [7, -2, -10],
            [-3, -3, 5],
        ]
    )
    keywords = pca.components_.T
    arrows = []
    for vector in vectors:
        keyword = max(
            range(len(keywords)),
            key=lambda i: np.dot(keywords[i], vector),
        )
        arrows.append((keywords[keyword], keyword_names[keyword]))

    for end_point, keyword_name in arrows:
        end_point = end_point / np.linalg.norm(end_point) * 20
        draw_arrow(end_point, keyword_name, view)

    if SHOW_PLOTS:
        vispy.app.run()
    if SAVE_PLOTS:
        view.camera.center = 5.2, 5.2, 1.4  # type: ignore
        view.camera.elevation = -10.5  # type: ignore
        view.camera.azimuth = -22.0  # type: ignore
        image = canvas.render((300, 100, 450, 480))
        io.write_png('pca.png', image)


def draw_arrow(end: NDArray, text: str, view):
    arrow = scene.visuals.Arrow(
        arrows=np.array([[0, 0, 0, *end]]),
        color='black',
        arrow_size=10,
        arrow_type='triangle_60',
        arrow_color='black',
        width=2,
        parent=view.scene,
    )
    line = scene.visuals.Line(
        pos=np.array([[0, 0, 0], end]), color='black', parent=view.scene, width=3
    )
    font_size = (600, 800)[text in ('nhl', 'zmaga', 'izrael')]
    text = scene.visuals.Text(
        text,
        pos=end * 1.1,
        color='black',
        parent=view.scene,
        font_size=font_size,
        face='Palatino',
    )
    view.add(arrow)
    view.add(line)
    view.add(text)


def tsne_task(X: NDArray, keywords: list[str], articles: list[Article]) -> None:
    """Transform the data using t-SNE and explain the transformed data."""
    kmeans = KMeans(n_clusters=5, random_state=1, n_init='auto').fit(X)

    # TSNE documentation recommends PCA as a pre-processing step
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=10, random_state=1).fit_transform(X)
    X_tsne = TSNE(n_components=2, random_state=1).fit_transform(X)

    # Sample for a lighter plot
    samples = np.random.choice(len(articles), 5000, replace=False)
    xs, ys = X_tsne[samples, 0], X_tsne[samples, 1]
    colors = kmeans.labels_[samples]

    plt.figure(figsize=(LINE_WIDTH, 0.8 * LINE_WIDTH))
    plt.scatter(xs, ys, s=10, alpha=0.5, c=colors, cmap='Pastel1', linewidth=0)
    plt.xlim(np.array([xs.min(), xs.max()]) * 1.2)
    plt.ylim(np.array([ys.min(), ys.max()]) * 1.2)
    plt.xticks([])
    plt.yticks([])

    # Annotate the clusters with the most common keywords
    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        cluster_center = np.mean(X_tsne[cluster_indices], axis=0)
        cluster_keywords = [
            keyword for i in cluster_indices for keyword in articles[i].keywords
        ]
        counter: Counter = Counter(cluster_keywords)
        most_common = counter.most_common(3)
        plt.text(
            cluster_center[0],
            cluster_center[1],
            ', '.join([keyword for keyword, _ in most_common]),
            fontsize=7,
            ha='center',
        )

    if SAVE_PLOTS:
        plt.savefig('tsne.pdf', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.tight_layout()
        plt.show()
    plt.close()


def main() -> None:
    # Plots configuration
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Palatino'

    articles = load_data('rtvslo_keywords.json')
    for a in articles:
        a.keywords = preprocess_keywords(a.keywords)
    X, keywords = create_features(articles)

    pca_task(X, keywords)
    # tsne_task(X, keywords, articles)


if __name__ == '__main__':
    main()
