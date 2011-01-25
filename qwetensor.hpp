#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace qwe {
    /**
     * Vector coordinates type.
     */
    typedef double vcoord_t;

    const unsigned short int default_dim = 3;

    /**
     * (2, 0)-tensor over 3-dimensinal euclidean space
     */
    template <int dim = default_dim, class T = vcoord_t>
    class Tensor {
    private:
        /**
         * Matrix type.
         */
        typedef std::vector<std::vector<T> > sto_t;

        sto_t storage;
    public:
        /**
         * Element index type.
         */
        typedef unsigned short int dim_t;

        /**
         * Forward iterator class used to traverse tensor elements by
         * rows.
         */
        class Iterator {
        protected:
            Tensor<dim, T> *tensor;
            dim_t row, col;

        public:
            typedef std::forward_iterator_tag iterator_category;
            typedef T value_type;
            typedef ptrdiff_t difference_type;
            typedef T pointer;
            typedef T& reference;

            Iterator(void)
            :tensor(0), row(0), col(0)
            {}

            Iterator(Tensor<dim, T>* t, dim_t r, dim_t c)
                :tensor(t), row(r), col(c)
            {}

            /**
             * Finish at [dim][0].
             */
            Iterator& operator ++(void)
            {
                col++;
                if (col == dim)
                {
                    col = 0;
                    row++;
                }
                return *this;
            }

            Iterator& operator ++(int)
            {
                return ++(*this);
            }

     
            bool operator ==(Iterator iter)
            {
                return (tensor == iter.tensor && row == iter.row && col == iter.col);
            }
            
            bool operator !=(Iterator iter)
            {
                return (tensor != iter.tensor || row != iter.row || col != iter.col);
            }

            Iterator& operator =(Iterator iter)
            {
                tensor = iter.tensor;
                row = iter.row;
                col = iter.col;
                return *this;
            }

            reference operator *(void)
            {
                return (*tensor)[row][col];
            }
        };

        class ColIterator {
        private:
            Tensor<dim, T> *tensor;
            dim_t row, col;
        public:
            typedef std::forward_iterator_tag iterator_category;
            typedef T value_type;
            typedef ptrdiff_t difference_type;
            typedef T pointer;
            typedef T& reference;

            ColIterator(void)
            :tensor(0), row(0), col(0)
            {}

            ColIterator(Tensor<dim, T>* t, dim_t r, dim_t c)
                :tensor(t), row(r), col(c)
            {}

            /**
             * Finish at [0][dim].
             */
            ColIterator& operator ++(void)
            {
                row++;
                if (row == dim)
                {
                    row = 0;
                    col++;
                }
                return *this;
            }

            ColIterator& operator ++(int)
            {
                return ++(*this);
            }

            bool operator ==(ColIterator iter)
            {
                return (tensor == iter.tensor && row == iter.row && col == iter.col);
            }
            
            bool operator !=(ColIterator iter)
            {
                return (tensor != iter.tensor || row != iter.row || col != iter.col);
            }

            ColIterator& operator =(ColIterator iter)
            {
                tensor = iter.tensor;
                row = iter.row;
                col = iter.col;
                return *this;
            }

            reference operator *(void)
            {
                return (*tensor)[row][col];
            }
        };

        Tensor (const sto_t &vex)
            :storage(vex)
        {}

        /**
         * Empty tensor.
         */
        Tensor (void)
        {
            storage.resize(dim);
            for (typename sto_t::iterator i = storage.begin(); i != storage.end(); i++)
                i->resize(dim);
        }

        std::vector<T>& operator [](const int &i) {
            return storage[i];
        }

        /**
         * Add two tensors.
         */
        const Tensor<dim, T> operator +(Tensor<dim, T> &T2)
        {
            Tensor<dim, T> nt;
            std::transform(this->hor_begin(), this->hor_end(), T2.hor_begin(), nt.hor_begin(), 
                           std::plus<T>());
            return nt;
        }

        /**
         * Multiply by scalar.
         */
        const Tensor<dim, T> operator *(const T &s)
        {
            Tensor<dim, T> nt;
            std::transform(this->hor_begin(), this->hor_end(), nt.hor_begin(), 
                           bind2nd(std::multiplies<T>(), s));
            return nt;
        }

        /**
         * Multiply by vector.
         */
        const std::vector<T> operator *(const std::vector<T> &v)
        {
            std::vector<T> nv;
            nv.resize(dim);
            for (dim_t i = 0; i < dim; i++)
                nv[i] = std::inner_product(this->row_begin(i), this->row_end(i), v.begin(), 0.0);
            return nv;
        }

        /**
         * Multiply by tensor.
         */
        const Tensor<dim, T> operator *(Tensor<dim, T> &T2)
        {
            Tensor<dim, T> nt;
            for (dim_t i = 0; i < dim; i++)
                for (dim_t j = 0; j < dim; j++)
                    nt[i][j] = std::inner_product(this->row_begin(i), this->row_end(i), 
                                                  T2.col_begin(j), 0.0);
            return nt;
        }
        
        /**
         * Double scalar tensor product.
         */
        const T operator ^(Tensor<dim, T> &T2)
        {
            return std::inner_product(this->hor_begin(), this->hor_end(),
                                      T2.vert_begin(), 0.0);
        }

        const Tensor<dim, T> transpose (void)
        {
            Tensor<dim, T> nt;
            std::copy(hor_begin(), hor_end(), nt.vert_begin());
            return nt;
        }


        /**
         * Iterate over all tensor elements row by row.
         */
        Iterator hor_begin(void)
        {
            return Iterator(this, 0, 0);
        }

        Iterator hor_end(void)
        {
            return Iterator(this, dim, 0);
        }

        /**
         * Iterate over one row only. Must be used with `row_end`
         * iterator called with the same row number.
         */
        Iterator row_begin(dim_t row)
        {
            return Iterator(this, row, 0);
        }

        Iterator row_end(dim_t row)
        {
            return Iterator(this, row + 1, 0);
        }

        /**
         * Iterate over all tensor elements column by column.
         */
        ColIterator vert_begin(void)
        {
            return ColIterator(this, 0, 0);
        }

        ColIterator vert_end(void)
        {
            return ColIterator(this, 0, dim);
        }

        /**
         * Iterate over one column only. Must be used with `col_end`
         * iterator called with the same row number.
         */
        ColIterator col_begin(dim_t col)
        {
            return ColIterator(this, 0, col);
        }

        ColIterator col_end(dim_t col)
        {
            return ColIterator(this, 0, col + 1);
        }

    };
}
