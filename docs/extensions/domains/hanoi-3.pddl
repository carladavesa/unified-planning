
;(define (problem hanoi3)
;  (:domain hanoi)
;  (:objects peg1 peg2 peg3 d1 d2 d3)
;  (:init
;   (smaller peg1 d1) (smaller peg1 d2) (smaller peg1 d3)
;   (smaller peg2 d1) (smaller peg2 d2) (smaller peg2 d3)
;   (smaller peg3 d1) (smaller peg3 d2) (smaller peg3 d3)
;   (smaller d2 d1) (smaller d3 d1) (smaller d3 d2)
;   (clear peg2) (clear peg3) (clear d1)
;   (on d3 peg1) (on d2 d3) (on d1 d2))
;  (:goal (and (on d3 peg3) (on d2 d3) (on d1 d2)))
;  )

(define (problem hanoi3)
    (:domain hanoi)

    (:objects
        peg1 peg2 peg3 - peg
        d1 d2 d3 - disc
    )


    (:init
        (= (top peg1) 5)
        (= (top peg2) 0)
        (= (top peg3) 0)
        (= (tower peg1) (array.mk (1 2 3 4 5)))
        (= (tower peg2) (array.mk (0 0 0 0 0)))
        (= (tower peg3) (array.mk (0 0 0 0 0)))
        ;(= (discs) (set.mk (1 2 3 4 5)))


    (smaller d1 peg1) (smaller d2 peg1) (smaller d3 peg1)
    (smaller d1 peg2) (smaller d2 peg2) (smaller d3 peg2)
    (smaller d1 peg3) (smaller d2 peg3) (smaller d3 peg3)
     (smaller d1 d2) (smaller d1 d3) (smaller d2 d3)
    (on d1 d2) (on d2 d3) (on d3 peg1)
    )

    (:goal (and
        (on d1 d2)
        (on d2 d3)
        (on d3 peg3)
        (= (top peg1) 5)
        (= (tower peg3) (array.mk (1 2 3 4 5)))
        )
    )
)