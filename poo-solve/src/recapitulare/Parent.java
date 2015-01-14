package recapitulare;

class Parent {
  public Parent() {
    System.out.println("Parent 0");
  }

  public Parent(int x) {
    System.out.println("Parent 1");
  }

  public static void main(String[] args) {
    new Child(2);
  }

}

class Child extends Parent {
  public Child(int x) {
    System.out.println("Child 1");
  }
}