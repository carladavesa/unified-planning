(define (problem sets-test2-problem)
    (:domain sets-test2)

    (:objects
        a b c d e - item
    )

    (:init
        (= (bag1) (set.mk (a b c)))
        (= (bag2) (set.mk (c d e)))
    )

    ;; Expected plan (2 steps):
    ;;   keep_common   → bag1 := {a,b,c} ∩ {c,d,e} = {c}
    ;;   take_complement → bag1 ⊆ bag2={c,d,e}, so bag1 := {c,d,e} \ {c} = {d,e}
    (:goal
        (= (bag1) (set.mk (d e)))
    )
)
