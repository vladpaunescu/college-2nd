package lab2;

public class Cricket {

  int lifeCycle;
  int yearsLeft;
  int N;

  public Cricket(int lifeCycle, int years, int N) {
    this.lifeCycle = lifeCycle;
    this.yearsLeft = years;
    this.N = N;
  }

  public void reproduce(Environment e) {
    yearsLeft = lifeCycle;
    N *= e.val_c;
  }

  public Triplet<Cricket> mutate(Environment e) {
    int increase = N * e.x_c;
    int same = N * e.y_c;
    int decrease = N - (increase + same);

    return Triplet.<Cricket> of(new Cricket(lifeCycle + 1, lifeCycle + 1, increase), new Cricket(
        lifeCycle, lifeCycle, same), new Cricket(lifeCycle - 1, lifeCycle - 1, decrease));
  }
}
