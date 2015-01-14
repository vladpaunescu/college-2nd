package recapitulare;

class Outer {
  int x = 10;

  class Inner extends Outer {
    int x = 15;

    void printFirst(int x) {
      System.out.println(this.x);
      System.out.println(x);
      System.out.println(super.x);
    }

    void printSecond() {
      System.out.println(this.x);
      System.out.println(x);
      System.out.println(super.x);
    }

    void printAll() {
      printFirst(20);
      printSecond();
    }
  }

  public static void main(String[] args) {
    new Outer().new Inner().printAll();
  }
}