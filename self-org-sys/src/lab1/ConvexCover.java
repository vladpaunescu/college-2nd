package lab1;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class ConvexCover {

  private static final Logger LOGGER = Logger.getLogger(ConvexCover.class.getName());

  private final int n;
  private final int matrix[][];
  private final List<Vertex> nodes;

  public ConvexCover(int n) {
    this.n = n;
    matrix = new int[n][n];
    nodes = new ArrayList<>();
    initializeData();
  }

  public void run() throws InterruptedException {
    LOGGER.info("Starting nodes...");
    for (Vertex node : nodes) {
      node.start();
    }

    LOGGER.info("Waiting for nodes to finish...");
    for (Vertex node : nodes) {
      node.join();
    }

    LOGGER.info("The vertex cover for " + n + " nodes is (adj matrix)");
    printMatrix(matrix);
  }

  private int getNextNode(int node) {
    if (node < n - 1) {
      return node + 1;
    }
    return 0;
  }

  private void initializeData() {
    for (int i = 0; i < n; ++i) {
      nodes.add(new Vertex(i, matrix));
      int nextNode = getNextNode(i);
      matrix[i][nextNode] = 1;
      matrix[nextNode][i] = 1;
    }
  }

  private void printMatrix(int[][] mat) {
    for (int i = 0; i < mat.length; ++i) {
      StringBuilder sb = new StringBuilder();
      for (int j = 0; j < mat[0].length; ++j) {
        sb.append(mat[i][j] + " ");
      }
      System.out.println(sb.toString());
    }
  }

  public static void main(String[] args) throws InterruptedException {
    int n = 5;
    ConvexCover cover = new ConvexCover(n);
    cover.run();
  }
}
