import pymysql
import pickle


def insert_quiz(fname):
    conn = pymysql.connect(host='snu.snu.ac.kr', user='root', password='elqlfoq123', db='quiz_generation',
                           charset='utf8')
    curs = conn.cursor()

    quizzes = pickle.load(open(fname, 'rb'))

    for quiz in quizzes[:10]:
        sql = "INSERT INTO quiz(title, sentence, answer"
        for i in range(len(quiz.distractors)):
            sql += ", distractor%d" % (i + 1)
        sql += ") VALUES(%s, %s, %s"
        for i in range(len(quiz.distractors)):
            sql += ", %s"
        sql += ");"
        tuple = (quiz.document, quiz.sentence, quiz.gap)
        for dist in quiz.distractors:
            tuple += (dist,)
        curs.execute(sql, tuple)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    insert_quiz('quiz/quiz.book.database.joint.pkl')