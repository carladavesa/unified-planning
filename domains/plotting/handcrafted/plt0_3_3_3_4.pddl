
(define (problem plt0_3_3_3_4)
(:domain plotting)
(:objects
	 n1 - number
	 n2 - number
	 n3 - number

	 c1 - colour
	 c2 - colour
	 c3 - colour

)
(:init
	(= (total-cost) 0)
	(coloured n1 n1 c1)
	(coloured n1 n2 c2)
	(coloured n1 n3 c2)
	(coloured n2 n1 c3)
	(coloured n2 n2 c3)
	(coloured n2 n3 c2)
	(coloured n3 n1 c2)
	(coloured n3 n2 c3)
	(coloured n3 n3 c1)

	(hand wildcard)
	(succ n2 n1)
	(pred n1 n2)
	(succ n3 n2)
	(pred n2 n3)
	(lt n1 n2)
	(lt n1 n3)
	(lt n2 n3)
	(gt n3 n2)
	(gt n3 n1)
	(gt n2 n1)

	(isfirstcolumn n1)
	(islastcolumn n3)
	(istoprow n1)
	(isbottomrow n3)
	(distance n1 n2 n1)
	(distance n1 n3 n2)
	(distance n2 n1 n1)
	(distance n2 n3 n1)
	(distance n3 n1 n2)
	(distance n3 n2 n1)

)

(:goal
    (exists (?x1 ?y1 ?x2 ?y2 ?x3 ?y3 ?x4 ?y4 - number)
	(and
	(or (not (= ?x1 ?x2)) (not (= ?y1 ?y2)))
	(or (not (= ?x1 ?x3)) (not (= ?y1 ?y3)))
	(or (not (= ?x1 ?x4)) (not (= ?y1 ?y4)))
	(or (not (= ?x2 ?x3)) (not (= ?y2 ?y3)))
	(or (not (= ?x2 ?x4)) (not (= ?y2 ?y4)))
	(or (not (= ?x3 ?x4)) (not (= ?y3 ?y4)))
	(forall (?x5 ?y5 - number) (or
    (and (= ?x1 ?x5) (= ?y1 ?y5))
    (and (= ?x2 ?x5) (= ?y2 ?y5))
    (and (= ?x3 ?x5) (= ?y3 ?y5))
    (and (= ?x4 ?x5) (= ?y4 ?y5))
	(coloured ?x5 ?y5 null)))))
)
(:metric minimize (total-cost))
)
