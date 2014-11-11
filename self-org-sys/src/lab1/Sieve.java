package lab1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

public class Sieve {

  private static final Logger LOGGER = Logger.getLogger(ConvexCover.class.getName());

  public final int n;
  private final AtomicBoolean[] primes;
  private final List<List<Integer>> divisors;
  private final List<PrimeCandidate> nodes;

  public Sieve(int n) {
    this.n = n;
    primes = new AtomicBoolean[n + 1];
    divisors = new ArrayList<>();
    for (int i = 1; i <= n; ++i) {
      primes[i] = new AtomicBoolean(true);
      divisors.add(Collections.synchronizedList(new LinkedList<>()));
    }

    nodes = new ArrayList<>();
    for (int i = 2; i <= n; ++i) {
      nodes.add(new PrimeCandidate(i, n, primes, divisors));
    }

  }

  public void run() throws InterruptedException {
    LOGGER.info("Starting nodes...");
    for (PrimeCandidate node : nodes) {
      node.start();
    }

    LOGGER.info("Waiting for nodes to finish...");
    for (PrimeCandidate node : nodes) {
      node.join();
    }

    printData();
  }

  private void printData() {
    LOGGER.info("The primes are:");
    StringBuilder sb = new StringBuilder();
    for (int i = 2; i <= n; ++i) {
      if (primes[i].get()) {
        sb.append(i + " ");
      }
    }
    System.out.println(sb.toString());
    LOGGER.info("The divisors are:");
    for (int i = 2; i <= n; ++i) {
      sb = new StringBuilder();
      sb.append(i + " " + divisors.get(i - 1).size() + ": ");
      for (Integer divisor : divisors.get(i - 1)) {
        sb.append(divisor + " ");
      }
      System.out.println(sb.toString());
    }
  }

  public static void main(String[] args) throws InterruptedException {
    int n = 1000;
    Sieve sieve = new Sieve(n);
    sieve.run();
  }

}
