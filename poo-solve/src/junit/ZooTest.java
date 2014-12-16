package junit;

import junit.framework.TestCase;

import org.junit.Assert;
import org.junit.Test;

public class ZooTest extends TestCase {

  Zoo zoo;

  @Override
  protected void setUp() throws Exception {
    zoo = new Zoo();
  }

  @Test
  public void testAddAnimal() {
    zoo.addAnimal(new Animal("cat"));
    assertEquals(1, zoo.size());
  }

  @Test
  public void testRemoveAnimal() {
    Animal a1 = new Animal("cat");
    Animal a2 = new Animal("dog");
    zoo.addAnimal(a1).addAnimal(a2);
    assertTrue(zoo.removeAnimal(a1));
  }

  @Test
  public void testAreAnimalsInZoo() {
    Animal a1 = new Animal("cat");
    Animal a2 = new Animal("dog");
    zoo.addAnimal(a1).addAnimal(a2);
    if (!zoo.areAnimals()) {
      Assert.fail("No animals");
    }
  }

  @Test
  public void testGetAnimals() {
    Animal a1 = new Animal("cat");
    Animal a2 = new Animal("dog");
    zoo.addAnimal(a1).addAnimal(a2);
    assertFalse(zoo.getAnimals().isEmpty());
  }
}
