plt.clf()
        networkx.draw(g,pos,node_color=colors)
        # ax.figure.canvas.draw()
        # ax.figure.canvas.flush_events()
        plt.pause(0.05)
        plt.show()