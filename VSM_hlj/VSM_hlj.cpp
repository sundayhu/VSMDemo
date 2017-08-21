// VSM_hlj.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <algorithm>
#include <hash_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
const int MAX = 3333;
//�ַ����ָ��
std::vector<std::string> split(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str += pattern;//��չ�ַ����Է������
	int size = str.size();

	for (int i = 0; i<size; i++)
	{
		pos = str.find(pattern, i);
		if (pos<size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}
int _tmain(int argc, _TCHAR* argv[])
{
	vector<hash_map<string, int >> words_num;//����Ƶ���ĵ�
	hash_map<string, int > term;//������
	vector<vector<string>> docs;//�ĵ�
	hash_map<string, int > stopWords;//ͣ�ô�
	string line;
	vector<string> words;
	/*��ȡͣ�ô��ļ�*/
	int count = 1;
	ifstream in_;
	in_.open("Chinese-StopWords.txt");
	if (!in_)
	{
		cout << "Chinese-StopWords.txt open error!" << endl;
		return 0;
	}
	while (!in_.eof())
	{
		getline(in_, line);
		//cout << line << endl;
		stopWords.insert(pair<string, int>(line, count++));
	}
	in_.close();
	//cout << stopWords.size() << endl;
	/*��ȡ�ĵ��ļ�*/
	int len = 0;
	count = 1;
	hash_map<string, int>::iterator s;
	hash_map<string, int>::iterator t;
	ifstream in;
	in.open("input.txt");
	if (!in)
	{
		cout << "input.txt open error!" << endl;
		return 0;
	}
	//int nnn = 0;
	while (!in.eof())
	{
		getline(in, line);
		words = split(line, " ");//�ÿո�ִ�
		len = words.size();
		vector<string> doc;
		hash_map<string, int > word_num;//�����ĵ��Ĵʼ���Ƶ
		for (int i = 1; i < len; i++)
		{
			//ȥ��ͣ�ô�
			s = stopWords.find(words[i]);//��ͣ�ô�
			if (s == stopWords.end())//����ͣ�ô�
			{
				doc.push_back(words[i]);
				//����term
				t = term.find(words[i]);
				if (t == term.end())//�����ڣ����µĴ���ӵ�term��
				{
					term.insert(pair<string, int>(words[i], count));
				}
				else//���ڣ����ôʵĴ�Ƶ+1
				{
					term[words[i]] = term[words[i]] + 1;
				}
				//�����ĵ��������Ƶ
				t = word_num.find(words[i]);
				if (t == word_num.end())//�����ڣ����µĴ���ӵ�word_num��
				{
					word_num.insert(pair<string, int>(words[i], count));
				}
				else//���ڣ����ôʵĴ�Ƶ+1
				{
					word_num[words[i]] = word_num[words[i]] + 1;
				}
			}
		}

		docs.push_back(doc);
		words_num.push_back(word_num);
		doc.clear();
		word_num.clear();
	}
	cout << "�ĵ�����Ϊ��"<<docs.size() << endl;
	cout << "term����Ϊ��"<<term.size() << endl;
	//cout << nnn << endl;
	in.close();
	//cout << terms.size() << endl;
	/*term��ά  --  ѡ����ִ����϶�Ĵ���Ϊterm*/
	t = term.begin();
	while (t != term.end())
	{
		if (t->second < 2)//ѡ����ִ���>=3�Ĵ���Ϊterm
		{
			s = t;
			t++;
			term.erase(s);
			continue;
		}
		t++;
	}
	cout <<"��ά��term����Ϊ��" <<term.size() << endl;
	/*Ȩ�ؼ��� TF IDF*/
	ofstream out_;
	out_.open("tf_idf.txt");
	//double matrix[MAX][MAX];//ջ���
	vector<vector<double>> TF_IDF;
	vector<hash_map<string, int >>::iterator w;
	double tf, idf,n,sum,tf_idf,d,j;
	d = docs.size();//���Ͽ��е��ļ�����
	for (int i = 0; i < d; i++)
	{
		vector<double> temp;
		for (t = term.begin(); t!=term.end(); t++)
		{
			//tf=n/sum
			n = words_num[i][t->first];//ÿ��term������ÿ��doc��Ĵ�Ƶ
			sum = docs[i].size();//ÿ��doc���ж��ٴ���
			tf = n / sum;
			//idf=log(d/j)
			j = 0;//
			for (w = words_num.begin(); w != words_num.end(); w++)
			{
				s = w->find(t->first);
				if (s != w->end())//����
				{
					j++;
				}
			}
			idf = log(d / j);
			tf_idf = tf*idf;
			out_ << tf_idf;
			out_ << " ";
			temp.push_back(tf_idf);
		}
		out_ << "\n";
		TF_IDF.push_back(temp);
		temp.clear();
	}
	out_.close();
	cout << "���ƶȴ���0.8���ĵ�(��������������)ͳ�����£�" << endl;
	//���ƶȼ���
	//sim(d,p)=d*p/(|d|*|p|)
	ofstream out;
	out.open("outs.txt");
	if (!out)
	{
		cout << "open out.txt error!" << endl;
		return 0;
	}
	int  term_num = term.size();
	double dp,m2,n2,sim;
	int sim_num = 0;
	for (int i = 0; i < d; i++)
	{
		m2 = 0.0;
		for (int k = 0; k < term_num; k++)
		{
			m2 += TF_IDF[i][k] * TF_IDF[i][k];//|d|
		}
		for (int j = i+1; j < d; j++)
		{
			dp = 0;
			n2 = 0.0;
			for (int k = 0; k < term_num; k++)
			{
				dp += TF_IDF[i][k] * TF_IDF[j][k];//d*p
			}
			for (int k = 0; k < term_num; k++)
			{
				n2 += TF_IDF[j][k] * TF_IDF[j][k];//|p|
			}
			sim = dp / (sqrt(m2)*sqrt(n2));
			out << i;
			out << " ";
			out << j;
			out << ":";
			out << sim;
			out << "\n";
			if (sim > 0.8)
			{
				cout << "�� " << i << " ƪ�ĵ���� " << j << " ƪ���ĵ�֮������ƶ�Ϊ��" << sim << endl;
			}
		}
	}
	cout << "���ƶȴ���0.8���ĵ�(��������������)�ܹ���" <<sim_num<< endl;
	out.close();
	return 0;
}

