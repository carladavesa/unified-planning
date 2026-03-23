;; Fifteen puzzle problem #24 from Korf's AIJ (27) paper.
;; Optimal solution cost = 54.

(define (problem korf24)
  (:domain strips-sliding-tile)
  (:objects t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15
	    p1 p2 p3 p4)
  (:init
   (= (total-cost) 0)
   (tile t1) (tile t2) (tile t3)
   (tile t4) (tile t5) (tile t6)
   (tile t7) (tile t8) (tile t9)
   (tile t10) (tile t11) (tile t12)
   (tile t13) (tile t14) (tile t15)
   (position p1) (position p2) (position p3) (position p4)
   (inc p1 p2) (inc p2 p3) (inc p3 p4)
   (dec p4 p3) (dec p3 p2) (dec p2 p1)

   ;; initial state: first (top) row, left-to-right
   (at t7 p1 p1)
   (at t3 p2 p1)
   (at t14 p3 p1)
   (at t13 p4 p1)
   ;; initial state: second row, left-to-right
   (at t4 p1 p2)
   (at t1 p2 p2)
   (at t10 p3 p2)
   (at t8 p4 p2)
   ;; initial state: third row, left-to-right
   (at t5 p1 p3)
   (at t12 p2 p3)
   (at t9 p3 p3)
   (at t11 p4 p3)
   ;; initial state: fourth (bottom) row, left-to-right
   (at t2 p1 p4)
   (at t15 p2 p4)
   (at t6 p3 p4)
   (blank p4 p4)
   )

  ;; standard goal state (but not mentioning blank)
  (:goal
   (and
    (at t1 p2 p1)
    (at t2 p3 p1)
    (at t3 p4 p1)
    (at t4 p1 p2)
    (at t5 p2 p2)
    (at t6 p3 p2)
    (at t7 p4 p2)
    (at t8 p1 p3)
    (at t9 p2 p3)
    (at t10 p3 p3)
    (at t11 p4 p3)
    (at t12 p1 p4)
    (at t13 p2 p4)
    (at t14 p3 p4)
    (at t15 p4 p4)))
  (:metric minimize (total-cost))
  )
