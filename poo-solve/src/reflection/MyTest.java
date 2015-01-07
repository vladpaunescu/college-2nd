package reflection;

public class MyTest {

  int a;

  @Before
  public void setUp() {
    a = 1;
  }

  @Test
  public boolean test1() {
    System.out.println("Testing for a == 1");
    return a == 1;
  }

  @Test
  public boolean test2() {
    return a > 2;
  }

  @After
  public void tearDown() {
    System.out.println("Tearing down");
    a = 0;
    System.out.println("a = " + a);
  }
}
