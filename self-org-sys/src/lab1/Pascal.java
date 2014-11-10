package lab1;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class Pascal {

  private static final Logger LOGGER = Logger.getLogger(Pascal.class.getName());

  public static void main(String[] args) throws InterruptedException {
    int n = 30;
    Element[][] triangle = new Element[n][n];
    for (int i = n - 1; i >= 0; --i) {
      for (int j = 0; j <= i; ++j) {
        List<Element> children = new ArrayList<>();
        if (i < n - 1) {
          children.add(triangle[i + 1][j]);
          children.add(triangle[i + 1][j + 1]);
        }

        int parents = 2;
        if (i == 0) {
          parents = 0;
        } else if (j == 0 || i == j) {
          parents = 1;
        }

        triangle[i][j] = Element.Builder
            .newBuilder()
              .withRow(i)
              .withColumn(j)
              .withMessageCount(parents)
              .withInitialSum((i == 0 && j == 0) ? 1 : 0)
              .addAllChildren(children)
              .build();
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        triangle[i][j].start();
      }
    }

    LOGGER.info("Waiting for threads to finish");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        triangle[i][j].join();
      }
    }

    LOGGER.info("Threads finished");

    for (int i = 0; i < n; ++i) {
      StringBuilder sb = new StringBuilder();
      for (int j = 0; j <= i; ++j) {
        sb.append(triangle[i][j].getSum()).append(" ");
      }
      System.out.println(sb.toString());
    }
  }
}
