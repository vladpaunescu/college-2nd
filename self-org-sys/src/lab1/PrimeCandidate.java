package lab1;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class PrimeCandidate extends Thread {

  private final int id;
  private final int n;
  private final AtomicBoolean[] primes;
  private final List<List<Integer>> divisors;

  public PrimeCandidate(int id, int n, AtomicBoolean[] isPrime, List<List<Integer>> divisors) {
    this.id = id;
    this.n = n;
    this.primes = isPrime;
    this.divisors = divisors;
  }

  @Override
  public void run() {
    for (int j = 2; id * j <= n; j++) {
      primes[id * j].set(false);
      divisors.get(id * j - 1).add(new Integer(id));
    }
  }
}
