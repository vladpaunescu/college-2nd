package lab3;

import java.util.ArrayList;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.logging.Logger;

public class Seller extends Thread {

  private static final Logger LOGGER = Logger.getLogger(Seller.class.getName());

  private int initialBudget;
  private int advertisingMoney;
  private double advertisingQuality;
  private int zeroProfitCount;
  private int minimumProductCount;

  private Seller(Builder builder) {
    this.initialBudget = builder.initialSum;
    this.advertisingMoney = builder.advertisingMoney;
    this.advertisingQuality = builder.advertisingQuality;
    this.zeroProfitCount = builder.zeroProfitCount;
    this.minimumProductCount = builder.minimumProductCount;
  }

  static class Builder {
    private int initialSum;
    private int advertisingMoney;
    private double advertisingQuality;
    private int zeroProfitCount;
    private int minimumProductCount;

    private Builder() {
    }

    public static Builder newBuilder() {
      return new Builder();
    }

    public Builder withInitialSum(int sum) {
      initialSum = sum;
      return this;
    }

    public Builder withAdvertisingQuality(double quality) {
      advertisingQuality = quality;
      return this;
    }

    public Builder withZeroProfitCount(int count) {
      zeroProfitCount = count;
      return this;
    }

    public Seller build() {
      return new Seller(this);
    }
  }
}

class Element extends Thread {

  private static final Logger LOGGER = Logger.getLogger(Element.class.getName());

  private final int row;
  private final int column;
  private final List<Element> children;
  private BlockingQueue<Integer> queue;
  private int messageCount;
  private int sum;

  private Element(Builder builder) {
    super(builder.row + " " + builder.column);
    row = builder.row;
    column = builder.column;
    children = builder.children;
    queue = new LinkedBlockingQueue<>();
    sum = builder.sum;
    messageCount = builder.messageCount;
  }

  static class Builder {

    private int row;
    private int column;
    private int messageCount;
    private int sum;
    private List<Element> children;

    private Builder() {
      children = new ArrayList<>();
    }

    public static Builder newBuilder() {
      return new Builder();
    }

    public Builder withRow(int row) {
      this.row = row;
      return this;
    }

    public Builder withColumn(int column) {
      this.column = column;
      return this;
    }

    public Builder withMessageCount(int count) {
      messageCount = count;
      return this;
    }

    public Builder addChild(Element child) {
      children.add(child);
      return this;
    }

    public Element build() {
      return new Element(this);
    }

    public Builder addAllChildren(List<Element> children) {
      this.children = children;
      return this;
    }

    public Builder withInitialSum(int sum) {
      this.sum = sum;
      return this;
    }
  }

  @Override
  public void run() {
    LOGGER.info("Waiting for data to be available at " + row + " " + column);
    try {
      while (messageCount > 0) {
        sum += queue.take();
        --messageCount;
        LOGGER.info(row + " " + column + " Messages left " + messageCount);
      }
      LOGGER.info(row + " " + column);
      for (Element child : children) {
        LOGGER.info("Notifiying child " + child.row + " " + child.column);
        child.addValue(sum);
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public synchronized void addValue(int value) {
    this.queue.add(value);
  }

  public int getSum() {
    return sum;
  }
}