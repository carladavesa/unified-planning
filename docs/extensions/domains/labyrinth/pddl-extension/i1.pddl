(define (problem labyrinth-i1)
    (:domain labyrinth)

    (:objects
        card_0  card_1  card_2  card_3
        card_4  card_5  card_6  card_7
        card_8  card_9  card_10 card_11
        card_12 card_13 card_14 card_15 - card
    )

    (:init
        (= (card_at) (array.mk ((card_0  card_6  card_7  card_3 )
                                (card_5  card_9  card_10 card_4 )
                                (card_8  card_13 card_14 card_11)
                                (card_12 card_1  card_2  card_15))))
        (= (robot_at) card_0)
        (connections card_0  W) (connections card_0  E)
        (connections card_6  S) (connections card_6  W)
        (connections card_7  N) (connections card_7  W) (connections card_7  S) (connections card_7  E)
        (connections card_3  N) (connections card_3  W) (connections card_3  E)
        (connections card_5  N) (connections card_5  W) (connections card_5  E)
        (connections card_9  S) (connections card_9  W)
        (connections card_10 N) (connections card_10 S)
        (connections card_4  S) (connections card_4  E)
        (connections card_8  N) (connections card_8  S) (connections card_8  E)
        (connections card_13 S) (connections card_13 W)
        (connections card_14 N) (connections card_14 S) (connections card_14 E)
        (connections card_11 W) (connections card_11 E)
        (connections card_12 N) (connections card_12 W)
        (connections card_1  N) (connections card_1  W) (connections card_1  S)
        (connections card_2  N) (connections card_2  S)
        (connections card_15 S) (connections card_15 W)
    )

    (:goal
        (and
            (= (robot_at) (read (card_at) 3 3))
            (connections (read (card_at) 3 3) S)
        )
    )
)
