import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    rank = np.loadtxt('output_15_linkgraph.txt')
    rank_sorted = rank[np.argsort(rank[:, 1])]  # sort matrix by row
    reversed_rank = np.flip(rank_sorted, 0)  # reverse the order of the matrix

    # distribution of PageRank scores
    sorted_values = reversed_rank[:, 1]
    fig, ax = plt.subplots()
    ax.plot(np.log10(sorted_values))
    ax.set_ylabel('Log of PageRank score')
    ax.set_xlabel('Rank order from top1')
    ax.set_title('Distribution of PageRank scores')

    # create top20 scores matrix
    top_20 = reversed_rank[: 20]
    top_20_ids = top_20[:, 0]
    top_20_values = top_20[:, 1]

    # create top20 old scores matrix
    rank_old = np.loadtxt('output_5_linkgraph.txt')
    top_20_values_old = []
    for i in top_20_ids:
        top_20_values_old.append(rank_old[np.where(rank_old[:, 0] == i)][0][1])
    old_values = np.array(top_20_values_old)

    # create id-url dictionary
    ids = np.loadtxt('id-url.txt', usecols=0)
    urls = np.loadtxt('id-url.txt', usecols=1, dtype=np.unicode_)
    id_url = dict(zip(ids, urls))
    top_20_urls = [id_url[int(idx)].split('//')[1] for idx in top_20_ids]

    # comparison of top 20 PageRank's sites after 5 and 15 iterations
    fig, ax = plt.subplots()
    # set width of bar
    barWidth = 0.3
    # Set position of bar on X axis
    r1 = np.arange(len(top_20_ids))
    r2 = [x + barWidth for x in r1]
    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(top_20_ids))], top_20_urls, rotation=70)
    # Make the plot
    ax.bar(r1, top_20_values, color='#7f6d5f', width=barWidth, edgecolor='white', label='15 iterations')
    ax.bar(r2, top_20_values_old, color='#557f2d', width=barWidth, edgecolor='white', label='5 iterations')
    # Create legend & Show graphic
    ax.legend()
    ax.set_ylabel('PageRank score')
    ax.set_xlabel('Top 20 pages')
    ax.set_title('Comparison of top 20 PageRank\'s sites after 5 and 15 iterations')
    plt.show()
