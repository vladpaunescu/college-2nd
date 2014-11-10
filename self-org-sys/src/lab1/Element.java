package lab1;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.logging.Logger;

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