package recapitulare;

public class MyList {

  private/* @ non_null */int a[];
  // @ invariant N >= 0 && N <= a.length;
  private int N;

  public MyList() {
    a = new int[100];
    N = 0;
  }

  public void addtoList(int element) {
    if (N < a.length) {
      a[N] = element;
      N++;
    }
  }

  public void sort() {
    insertSort(a);
  }

  /*
   * Pre and post conditions
   * 
   * pre: method parameter a != null
   * 
   * post: post - \forall int i; 0 <= i && i < A.length-1; A[i] <= A[i+1] States
   * that array A[0..A.length-1] is sorted in ascending order.
   * 
   * Initialization, Maintenance and Termination
   * ===========================================
   * 
   * 
   * Outer loop - iterate over each array element
   * 
   * loop_invariant (\forall int k; 0<=k && k <= j-1; A[k] <= A[j-1]); for( j=1;
   * j<A.length; j++) {
   * 
   * Initialization: Prior to first iteration of for statement: A[0..0] <= A[0];
   * A[0] maximum of A[0..0].
   * 
   * Maintenance: At start and end of each iteration of for statement: A[0..j-1]
   * <= A[j-1]; A[j-1] maximum of A[0..j-1].
   * 
   * Termination: j=A.length, by loop invariant A[0..j-1] <= A[j-1]; A[j-1]
   * maximum of A[0..j-1].
   * 
   * 
   * 
   * Inner loop - shifts larger sorted elements to right
   * 
   * loop_invariant (\forall int k; 0<=k && k < i; A[k] <= A[k+1]);
   * 
   * while (i > -1 && A[i] > key) { Initialization: Prior to first iteration of
   * while statement: A[0..i] is sorted. When i=0, range of k, 0<=k && k < 0, is
   * empty so is sorted.
   * 
   * Maintenance: At start and end of each iteration of while statement: A[0..i]
   * is sorted.
   * 
   * Termination: Each iteration i=i-1, until i=-1. A[0..-1] is empty hence
   * sorted.
   */

  /*
   * @ requires a != null;
   * 
   * @ modifies a[*];
   * 
   * @ ensures (\forall int i; 0 <= i && i < a.length-1; a[i] <= a[i+1]);
   */
  private void insertSort(int[] a) {
    int n = a.length;

    // @ loop_invariant (\forall int k; 0<=k && k <= j-1; A[k] <= A[j-1]);
    for (int i = 1; i < n; i++) {
      int x = a[i];
      int j = i - 1;

      // @ loop_invariant (\forall int k; 0<=k && k < i; A[k] <= A[k+1]);
      while (j >= 0 && a[j] > x) {
        a[j + 1] = a[j];
        j = j - 1;
      }
      a[j + 1] = x;
    }
  }

  public static void main(String[] args) {
    MyList list = new MyList();
    list.addtoList(4);
    list.addtoList(2);
    list.addtoList(1);
    list.addtoList(3);
    list.sort();
  }
}