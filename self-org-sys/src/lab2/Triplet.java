package lab2;

public class Triplet<T> {
  T first;
  T second;
  T third;

  private Triplet(T t1, T t2, T t3) {
    first = t1;
    second = t2;
    third = t3;
  }

  public static <T> Triplet<T> of(T t1, T t2, T t3) {
    return new Triplet<T>(t1, t2, t3);
  }
}
