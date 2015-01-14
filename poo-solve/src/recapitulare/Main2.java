package recapitulare;

public class Main2 {
  private static Main2 Instance = new Main2();

  public static Main2 getInstance() {
    return Instance;
  }

  public void print() {
    System.out.println(this.getClass());
  }

  // Application Entry Point
  public static void main(String[] Params) {
    Main2.getInstance().print();
    new Main2().print();
  }
}