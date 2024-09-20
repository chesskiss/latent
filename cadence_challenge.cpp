#include <iostream>
#include <cassert>
#include <memory>
#include <string> 

using namespace std;

class A
{
    unsigned m_in;
public:
    A (unsigned in) { m_in = f1(in); }
    unsigned mIn() const { return m_in; }
    unsigned f1(unsigned in) const { return (in % 10 * 10) + (in % 100 / 10); }
    virtual unsigned key() { return 56; }
    virtual unsigned key() const { return 78; }
    virtual bool check() { //V
        return key() == mIn(); }
    virtual bool check() const { 
        return key() == f1(m_in); }
    virtual bool check(unsigned toCheck) const { 
                std::cout << "key= "<<f1(key())<<std::endl;
        std::cout << "toCheck= "<<toCheck<<std::endl;
        return toCheck == f1(key()); }
    virtual bool check(unsigned toCheck) { //V

        return toCheck == f1(key()); }
};

class B : public A
{
    unsigned m_in;
public:
    B (unsigned in) : A(in) { m_in = f1(in); }
    unsigned f1(unsigned in) const { 
        // std::cout<<"in="<<in<<endl;
        return 2 * (in % 10 * 10) + (in % 100 / 10); }
    unsigned key() const override { return 84; }
    unsigned key() override { return 12; }
    bool check() const override {   

        return key() == 2 * f1(mIn()); }
    bool check() override { //V
        
        return key() == 2 * m_in; }
    bool check(unsigned toCheck) const override { //V
        return toCheck == f1(key()); }
    bool check(unsigned toCheck) override { //V
                
        return toCheck == f1(key()); }
};

class CadenceChallenge
{
public:
    bool run(string input)
    {
        std::shared_ptr<A> a = std::make_shared<A>(parse(input));
        if (!a->check()){
            std::cout << "a false"<<std::endl;
            return false;
        }
            
        std::shared_ptr<B> b = std::make_shared<B>(parse(input));
        if (!b->check()){
            std::cout << "b false"<<std::endl;
            return false;
        }

        A* a2 = b.get();
        if (!a2->check(parse(input))){
            
            return false;
        }//V
        const B* b2 = b.get();
        if (!b2->check(parse(input)))//V
            return false;
////////////////////////////
        A a3 = *b.get();
        if (!a3.check(parse(input) + a->mIn()))
            return false;
////////////////////V
        const A a4 = *b.get();
        std::cout<<b->mIn();
        if (!a4.check(parse(input) + b->mIn()))
            return false;
            std::cout<<"checkpoint"<<endl;

        return input.empty();
    }

private:
    unsigned parse(string& input)
    {
        unsigned ret = std::stoul(input.substr(input.size() - 2));
        input.pop_back();
        input.pop_back();
        return ret;
    }
};

int main()
{
    bool pass = false;
    int start = -10;
    // for (int i = start; i>-1000000; i--){
    //     if ((0-i)%100 != 65){
    //         std::string input = std::to_string(i);        
    std::string input = "810988416065";
            if (CadenceChallenge().run(input)){
                std::cout << "WELL DONE! YOU'VE ENTERED THE LOTTERY!" << std::endl;
                pass = true;
                // break;
            }
            else{
                // std::cout << input << std::endl;
                std::cout << "Oh no! You've failed our challenge. Maybe next time..." << std::endl;
            }
    //     }
    // }
    std::cout << "passed? " << pass;
    return 0;
}
