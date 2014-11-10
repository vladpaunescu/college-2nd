package lab1;

public class Vertex extends Thread {

  private final int id;
  private final int[][] matrix;

  public Vertex(int id, int[][] matrix) {
    this.id = id;
    this.matrix = matrix;
  }

  @Override
  public void run() {
    if (id != 0) {
      System.out.println("Vertex " + id + " updating edges to root.");
      matrix[id][0] = matrix[0][id] = 1;
    }
  }
}
