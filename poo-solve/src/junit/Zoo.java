package junit;

import java.util.ArrayList;
import java.util.List;

public class Zoo {

  List<Animal> animals;

  public Zoo() {
    System.out.println("New zoo");
    animals = new ArrayList<>();
  }

  public Zoo addAnimal(Animal a) {
    // animals.add(a);
    return this;
  }

  public boolean removeAnimal(Animal a) {
    return false;
    // return animals.remove(a);
  }

  public boolean areAnimals() {
    return false;
    // return animals.size() > 0;
  }

  public List<Animal> getAnimals() {
    return null;
    // return animals;
  }

  public int size() {
    return 0;
    // return animals.size();
  }

}
