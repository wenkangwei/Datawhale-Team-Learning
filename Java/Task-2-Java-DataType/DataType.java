/* Java Review Task 2: DataType
 * Author: Wenkang Wei
 * This file is to test the 
 *
 * */

// Test enumerate class
enum Color {Orange, Black, White, Brown};
enum Gender {Male, Female};

class Cat{

/*Test different data types:
 * 
 * */
static String category = "mammal";
int num_legs = 4;
char id = 'a';
boolean has_wing = false;
byte body_temperature = 37;
float tall = 20.5f; // note: if using float tall=20.5,  20.5 is default to be double, then it is converted to be float
double length = 40.1; 
String name = "Mike";
long num_teeth = 32;
Color skin_color = Color.Orange;
final Gender gender = Gender.Male; //constant variable, it can not be changed anymore


public Cat(String name){
	this.name = name;

}

public float speed(){
	float speed = 10; //m/s
	return speed;
}

public void printInfo(){

System.out.println("Name: "+this.name);
System.out.println("Gender: "+this.gender);
System.out.println("#_legs: "+this.num_legs);
System.out.println("id: "+this.id);
System.out.println("Color: "+this.skin_color);
System.out.println("#Teeth: "+this.num_teeth);
System.out.println("Tall: "+this.tall + "Length:"+ this.length+"Body Temperature: "+ this.body_temperature);

}

}


class House<T>{

// note: in Python, when defining an array:  array = [...]

// in Java/C/C++,  T[] array={....},    T[] indicates this is an array and use bracket {} to initialize

private int size;
private int pt=0;
private T[] array;
public House(int size){
	this.size = size;
	this.array =  (T[])new Object[size];
}

public int size(){
	return this.size;

}

public T At(int i){
return this.array[i];
}

public void add(T e){
	this.array[this.pt] = e;

	if(this.pt< this.size) {
		this.pt+=1;
		this.size += 1;
	}
}

public void remove(T e){
	if(this.pt>=0){
	this.array[this.pt] = null;
	this.pt -= 1;
	this.size -= 1;
	}

}

}

public class DataType{
// define a class called DataType
public static void main(String[] args){
//main function entry. JVM will run this function first
System.out.println("Testing datatype...");
int arr[];
Cat c = new Cat("Mike");
c.printInfo();

House<Cat> h = new House<Cat>(10);
String[] ls = {"Mike","Jack","Nathan","William","James"};

for(int i =0; i< ls.length; i++){
	h.add(new Cat(ls[i]));
}

for(int i =0; i<ls.length; i++){
	System.out.println(h.At(i).name);
}


}

}
