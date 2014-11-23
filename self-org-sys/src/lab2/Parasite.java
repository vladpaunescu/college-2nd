package lab2;

public class Parasite {

  int lifeCycle;
  int yearsLeft;
  int M;
  int cyclesLeft;

  public Parasite(int lifeCycle, int years, int M, int cyclesLeft) {
    this.lifeCycle = lifeCycle;
    this.yearsLeft = years;
    this.M = M;
    this.cyclesLeft = cyclesLeft;
  }

  public void reproduce(Environment e) {
    yearsLeft = lifeCycle;
    M *= e.val_p;
    cyclesLeft--;

  }

  public Triplet<Parasite> mutate(Environment e) {
    int increase = M * e.x_p;
    int same = M * e.y_p;
    int decrease = M - (increase + same);

    return Triplet.<Parasite> of(new Parasite(lifeCycle + 1, lifeCycle + 1, increase, e.nc),
        new Parasite(lifeCycle, lifeCycle, same, e.nc), new Parasite(lifeCycle - 1, lifeCycle - 1,
            decrease, e.nc));
  }

}
